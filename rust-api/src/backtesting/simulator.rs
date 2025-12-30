//! Backtest Simulator
//!
//! Validate profitability of EV > threshold strategy using historical data

use super::metrics::{calculate_metrics, BacktestMetrics};
use super::synthetic::SyntheticOddsGenerator;
use crate::core::kelly::KellyCalculator;
use crate::data::{load_exacta_odds, IndexedRaceData, RaceKey};
use crate::models::RacerEntry;
use crate::predictor::{FallbackPredictor, Predictor};
use crate::{ExactaPrediction, PositionProb};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Individual bet record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetRecord {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub first: u8,
    pub second: u8,
    pub probability: f64,
    pub odds: f64,
    pub expected_value: f64,
    pub stake: i64,
    pub actual_first: u8,
    pub actual_second: u8,
    pub won: bool,
    pub profit: i64,
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub bets: Vec<BetRecord>,
    pub total_races: usize,
    pub races_with_bets: usize,
    pub total_stake: i64,
    pub total_payout: i64,
    pub metrics: Option<BacktestMetrics>,
}

impl BacktestResult {
    pub fn new() -> Self {
        Self {
            bets: Vec::new(),
            total_races: 0,
            races_with_bets: 0,
            total_stake: 0,
            total_payout: 0,
            metrics: None,
        }
    }

    pub fn total_profit(&self) -> i64 {
        self.total_payout - self.total_stake
    }

    pub fn roi(&self) -> f64 {
        if self.total_stake == 0 {
            0.0
        } else {
            self.total_profit() as f64 / self.total_stake as f64
        }
    }

    pub fn finalize(&mut self) {
        self.metrics = Some(calculate_metrics(&self.bets, self.total_stake));
    }
}

impl Default for BacktestResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Indexed results data with O(1) lookups
///
/// Pre-loads all results into memory and indexes by (date, stadium_code, race_no)
pub struct IndexedResultsData {
    /// Results indexed by race key: (1st place boat, 2nd place boat)
    results: HashMap<RaceKey, (u8, u8)>,
}

impl IndexedResultsData {
    /// Load and index all results from CSV
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, PolarsError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(csv_path.as_ref().to_path_buf()))?
            .finish()?;

        let date_col = df.column("date")?.i64()?;
        let stadium_col = df.column("stadium_code")?.i64()?;
        let race_col = df.column("race_no")?.i64()?;
        let boat_col = df.column("boat_no")?.i64()?;
        let rank_col = df.column("rank")?.i64()?;

        // First pass: collect all entries by race
        let mut race_entries: HashMap<RaceKey, Vec<(u8, u8)>> = HashMap::new();

        for i in 0..df.height() {
            if let (Some(date), Some(stadium), Some(race), Some(boat), Some(rank)) = (
                date_col.get(i),
                stadium_col.get(i),
                race_col.get(i),
                boat_col.get(i),
                rank_col.get(i),
            ) {
                let key = (date as u32, stadium as u8, race as u8);
                race_entries
                    .entry(key)
                    .or_default()
                    .push((boat as u8, rank as u8));
            }
        }

        // Second pass: extract 1st and 2nd place for each race
        let mut results: HashMap<RaceKey, (u8, u8)> = HashMap::new();

        for (key, entries) in race_entries {
            let mut first: Option<u8> = None;
            let mut second: Option<u8> = None;

            for (boat, rank) in entries {
                if rank == 1 {
                    first = Some(boat);
                } else if rank == 2 {
                    second = Some(boat);
                }
            }

            if let (Some(f), Some(s)) = (first, second) {
                results.insert(key, (f, s));
            }
        }

        Ok(Self { results })
    }

    /// Get race result (1st and 2nd place boats) - O(1)
    pub fn get_race_result(&self, date: u32, stadium_code: u8, race_no: u8) -> Option<(u8, u8)> {
        self.results.get(&(date, stadium_code, race_no)).copied()
    }

    /// Total number of races with results
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

/// Backtest simulator configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub ev_threshold: f64,
    pub stake: i64,
    pub max_bets_per_race: usize,
    pub use_kelly: bool,
    pub kelly_multiplier: f64,
    pub test_start_date: Option<u32>,
    /// Path to ONNX models directory (if None, uses fallback predictor)
    pub model_dir: Option<PathBuf>,
    /// Use synthetic odds when real odds are not available
    pub use_synthetic_odds: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            ev_threshold: 1.0,
            stake: 100,
            max_bets_per_race: 3,
            use_kelly: false,
            kelly_multiplier: 0.25,
            test_start_date: Some(20240701), // Second half of 2024
            model_dir: None,
            use_synthetic_odds: false,
        }
    }
}

/// Unified predictor that can use either ONNX or fallback
enum UnifiedPredictor {
    Onnx(Predictor),
    Fallback(FallbackPredictor),
}

impl UnifiedPredictor {
    fn predict_positions(&mut self, entries: &[RacerEntry]) -> Vec<PositionProb> {
        match self {
            UnifiedPredictor::Onnx(p) => p.predict_positions(entries).unwrap_or_else(|e| {
                eprintln!("ONNX prediction failed: {}, using fallback", e);
                FallbackPredictor::new().predict_positions(entries)
            }),
            UnifiedPredictor::Fallback(p) => p.predict_positions(entries),
        }
    }

    fn calculate_exacta_probs(&self, position_probs: &[PositionProb]) -> Vec<ExactaPrediction> {
        match self {
            UnifiedPredictor::Onnx(p) => p.calculate_exacta_probs(position_probs),
            UnifiedPredictor::Fallback(p) => p.calculate_exacta_probs(position_probs),
        }
    }
}

/// Backtest simulator
pub struct BacktestSimulator {
    pub config: BacktestConfig,
    predictor: UnifiedPredictor,
    kelly: Option<KellyCalculator>,
    synthetic_odds: Option<SyntheticOddsGenerator>,
}

impl BacktestSimulator {
    /// Create a new backtest simulator
    pub fn new(config: BacktestConfig) -> Self {
        let kelly = if config.use_kelly {
            Some(KellyCalculator::new(
                100_000, // default bankroll
                config.kelly_multiplier,
                100,  // min stake
                0.10, // max stake pct
                0.30, // max total exposure
            ))
        } else {
            None
        };

        // Try to load ONNX models if path provided
        let predictor = if let Some(ref model_dir) = config.model_dir {
            match Predictor::new(model_dir) {
                Ok(p) => {
                    eprintln!("Using ONNX predictor from {:?}", model_dir);
                    UnifiedPredictor::Onnx(p)
                }
                Err(e) => {
                    eprintln!("Failed to load ONNX models: {}, using fallback", e);
                    UnifiedPredictor::Fallback(FallbackPredictor::new())
                }
            }
        } else {
            eprintln!("No model directory specified, using fallback predictor");
            UnifiedPredictor::Fallback(FallbackPredictor::new())
        };

        // Create synthetic odds generator if enabled
        let synthetic_odds = if config.use_synthetic_odds {
            eprintln!("Synthetic odds enabled (25% margin)");
            Some(SyntheticOddsGenerator::default())
        } else {
            None
        };

        Self {
            config,
            predictor,
            kelly,
            synthetic_odds,
        }
    }

    /// Run backtest on historical data
    ///
    /// Uses pre-indexed data structures for O(1) lookups instead of O(n) filtering.
    pub fn run<P: AsRef<Path>>(
        &mut self,
        programs_path: P,
        results_path: P,
        odds_dir: Option<&Path>,
    ) -> Result<BacktestResult, PolarsError> {
        // Load and index all data upfront
        eprintln!("Loading program data...");
        let race_data = IndexedRaceData::load(&programs_path)?;
        eprintln!("Loaded {} races from programs", race_data.len());

        eprintln!("Loading results data...");
        let results_data = IndexedResultsData::load(&results_path)?;
        eprintln!("Loaded {} races from results", results_data.len());

        let mut result = BacktestResult::new();

        // Iterate directly over all races (already indexed)
        for ((date, stadium_code, race_no), entries) in race_data.iter() {
            // Apply test start date filter
            if let Some(start) = self.config.test_start_date {
                if *date < start {
                    continue;
                }
            }

            if entries.len() != 6 {
                continue;
            }

            result.total_races += 1;

            // Get actual result - O(1) lookup
            let (actual_first, actual_second) =
                match results_data.get_race_result(*date, *stadium_code, *race_no) {
                    Some(r) => r,
                    None => continue,
                };

            // Load odds if available, or use synthetic odds
            let real_odds: Option<HashMap<(u8, u8), f64>> =
                odds_dir.and_then(|dir| load_exacta_odds(dir, *date, *stadium_code, *race_no));

            let odds = match real_odds {
                Some(o) => o,
                None => {
                    // Use synthetic odds if enabled, otherwise skip
                    if let Some(ref synth) = self.synthetic_odds {
                        synth.get_all_odds()
                    } else {
                        continue;
                    }
                }
            };

            // Run prediction
            let racer_entries: Vec<_> = entries.iter().map(|e| e.to_racer_entry()).collect();
            let position_probs = self.predictor.predict_positions(&racer_entries);
            let exacta_probs = self.predictor.calculate_exacta_probs(&position_probs);

            // Calculate expected values and filter
            let mut value_bets: Vec<_> = exacta_probs
                .iter()
                .filter_map(|pred| {
                    let key = (pred.first, pred.second);
                    odds.get(&key).map(|&o| {
                        let ev = pred.probability * o;
                        (pred.first, pred.second, pred.probability, o, ev)
                    })
                })
                .filter(|(_, _, _, _, ev)| *ev > self.config.ev_threshold)
                .collect();

            if value_bets.is_empty() {
                continue;
            }

            // Sort by EV descending
            value_bets.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

            // Limit to max bets per race
            value_bets.truncate(self.config.max_bets_per_race);

            result.races_with_bets += 1;

            // Execute bets
            for (first, second, probability, bet_odds, ev) in value_bets {
                let stake = self.calculate_stake(probability, bet_odds);
                let won = first == actual_first && second == actual_second;
                let payout = if won {
                    (bet_odds * stake as f64) as i64
                } else {
                    0
                };
                let profit = payout - stake;

                let record = BetRecord {
                    date: *date,
                    stadium_code: *stadium_code,
                    race_no: *race_no,
                    first,
                    second,
                    probability,
                    odds: bet_odds,
                    expected_value: ev,
                    stake,
                    actual_first,
                    actual_second,
                    won,
                    profit,
                };

                result.bets.push(record);
                result.total_stake += stake;
                result.total_payout += payout;
            }
        }

        result.finalize();
        Ok(result)
    }

    /// Calculate stake amount
    fn calculate_stake(&self, probability: f64, odds: f64) -> i64 {
        if let Some(ref kelly) = self.kelly {
            let sizing = kelly.calculate_single(probability, odds);
            sizing.stake.max(self.config.stake)
        } else {
            self.config.stake
        }
    }

    /// Print summary of backtest result
    pub fn print_summary(&self, result: &BacktestResult) {
        println!("\n{}", "=".repeat(60));
        println!("BACKTEST RESULTS");
        println!("{}", "=".repeat(60));
        println!("EV Threshold: {:.2}", self.config.ev_threshold);
        println!("Stake per bet: {}", self.config.stake);
        println!("Max bets per race: {}", self.config.max_bets_per_race);
        println!("{}", "-".repeat(60));
        println!("Total races: {}", result.total_races);
        println!("Races with bets: {}", result.races_with_bets);
        println!("Total bets: {}", result.bets.len());
        println!(
            "Winning bets: {}",
            result.bets.iter().filter(|b| b.won).count()
        );
        println!("{}", "-".repeat(60));
        println!("Total stake: {}", result.total_stake);
        println!("Total payout: {}", result.total_payout);
        println!("Total profit: {}", result.total_profit());
        println!("ROI: {:.1}%", result.roi() * 100.0);

        if let Some(ref metrics) = result.metrics {
            println!("{}", "-".repeat(60));
            println!("Hit rate: {:.1}%", metrics.hit_rate * 100.0);
            println!("Average EV: {:.2}", metrics.avg_ev);
            println!("Profit factor: {:.2}", metrics.profit_factor);
            println!("Max drawdown: {}", metrics.max_drawdown);
        }

        println!("{}", "=".repeat(60));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_result_new() {
        let result = BacktestResult::new();
        assert!(result.bets.is_empty());
        assert_eq!(result.total_races, 0);
        assert_eq!(result.total_stake, 0);
    }

    #[test]
    fn test_backtest_result_roi() {
        let mut result = BacktestResult::new();
        result.total_stake = 1000;
        result.total_payout = 1200;
        assert!((result.roi() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_backtest_result_roi_zero_stake() {
        let result = BacktestResult::new();
        assert_eq!(result.roi(), 0.0);
    }

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert!((config.ev_threshold - 1.0).abs() < 0.01);
        assert_eq!(config.stake, 100);
        assert_eq!(config.max_bets_per_race, 3);
        assert!(!config.use_kelly);
    }

    #[test]
    fn test_backtest_simulator_new() {
        let config = BacktestConfig::default();
        let simulator = BacktestSimulator::new(config);
        assert!(simulator.kelly.is_none());
    }

    #[test]
    fn test_backtest_simulator_with_kelly() {
        let config = BacktestConfig {
            use_kelly: true,
            ..Default::default()
        };
        let simulator = BacktestSimulator::new(config);
        assert!(simulator.kelly.is_some());
    }

    #[test]
    fn test_bet_record_serialization() {
        let record = BetRecord {
            date: 20240115,
            stadium_code: 23,
            race_no: 1,
            first: 1,
            second: 2,
            probability: 0.10,
            odds: 8.0,
            expected_value: 0.80,
            stake: 100,
            actual_first: 1,
            actual_second: 2,
            won: true,
            profit: 700,
        };

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: BetRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.date, 20240115);
        assert!(deserialized.won);
        assert_eq!(deserialized.profit, 700);
    }
}
