//! Backtest Simulator
//!
//! Validate profitability of EV > threshold strategy using historical data

use super::metrics::{calculate_metrics, BacktestMetrics};
use super::synthetic::SyntheticOddsGenerator;
use crate::core::kelly::KellyCalculator;
use crate::data::{load_exacta_odds, IndexedRaceData, PreDeadlineOdds, RaceKey, SnapshotIndex};
use crate::models::RacerEntry;
use crate::predictor::UnifiedPredictor;
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
    /// Lookahead-free mode only: races skipped because no snapshot was captured
    /// strictly before the deadline (and no synthetic fallback was enabled).
    pub races_excluded_no_predeadline_odds: usize,
    /// Lookahead-free mode only: races with no pre-deadline snapshot that fell
    /// back to synthetic odds. Kept distinct from real-odds bets so synthetic
    /// ROI is never silently reported as real pre-deadline ROI.
    pub races_fell_back_to_synthetic: usize,
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
            races_excluded_no_predeadline_odds: 0,
            races_fell_back_to_synthetic: 0,
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
    /// Exhibition times indexed by race key: [boat1_time, boat2_time, ..., boat6_time]
    exhibition_times: HashMap<RaceKey, [f64; 6]>,
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
        let exhibition_col = df.column("exhibition_time")?.f64()?;

        // First pass: collect all entries by race
        let mut race_entries: HashMap<RaceKey, Vec<(u8, u8, f64)>> = HashMap::new();

        for i in 0..df.height() {
            if let (Some(date), Some(stadium), Some(race), Some(boat), Some(rank)) = (
                date_col.get(i),
                stadium_col.get(i),
                race_col.get(i),
                boat_col.get(i),
                rank_col.get(i),
            ) {
                let exhibition = exhibition_col.get(i).unwrap_or(6.80);
                let key = (date as u32, stadium as u8, race as u8);
                race_entries
                    .entry(key)
                    .or_default()
                    .push((boat as u8, rank as u8, exhibition));
            }
        }

        // Second pass: extract 1st/2nd place and exhibition times for each race
        let mut results: HashMap<RaceKey, (u8, u8)> = HashMap::new();
        let mut exhibition_times: HashMap<RaceKey, [f64; 6]> = HashMap::new();

        for (key, entries) in race_entries {
            let mut first: Option<u8> = None;
            let mut second: Option<u8> = None;
            let mut times = [6.80; 6]; // Default exhibition time

            for (boat, rank, exhibition) in entries {
                if rank == 1 {
                    first = Some(boat);
                } else if rank == 2 {
                    second = Some(boat);
                }
                // Store exhibition time by boat index (boat_no - 1)
                if boat >= 1 && boat <= 6 {
                    times[(boat - 1) as usize] = exhibition;
                }
            }

            if let (Some(f), Some(s)) = (first, second) {
                results.insert(key, (f, s));
                exhibition_times.insert(key, times);
            }
        }

        Ok(Self {
            results,
            exhibition_times,
        })
    }

    /// Get race result (1st and 2nd place boats) - O(1)
    pub fn get_race_result(&self, date: u32, stadium_code: u8, race_no: u8) -> Option<(u8, u8)> {
        self.results.get(&(date, stadium_code, race_no)).copied()
    }

    /// Get exhibition times for all 6 boats - O(1)
    pub fn get_exhibition_times(
        &self,
        date: u32,
        stadium_code: u8,
        race_no: u8,
    ) -> Option<[f64; 6]> {
        self.exhibition_times
            .get(&(date, stadium_code, race_no))
            .copied()
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

/// Indexed weather data with O(1) lookups
pub struct IndexedWeatherData {
    /// Weather data indexed by race key: (weather, wind_direction, wind_speed, wave_height)
    weather: HashMap<RaceKey, (String, String, f64, f64)>,
}

impl IndexedWeatherData {
    /// Load and index weather data from results_races.csv
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, PolarsError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(csv_path.as_ref().to_path_buf()))?
            .finish()?;

        let date_col = df.column("date")?.i64()?;
        let stadium_col = df.column("stadium_code")?.i64()?;
        let race_col = df.column("race_no")?.i64()?;
        let weather_col = df.column("weather")?.str()?;
        let wind_dir_col = df.column("wind_direction")?.str()?;
        let wind_speed_col = df.column("wind_speed")?.i64()?;
        let wave_height_col = df.column("wave_height")?.i64()?;

        let mut weather_map: HashMap<RaceKey, (String, String, f64, f64)> = HashMap::new();

        for i in 0..df.height() {
            if let (Some(date), Some(stadium), Some(race)) =
                (date_col.get(i), stadium_col.get(i), race_col.get(i))
            {
                let weather = weather_col.get(i).unwrap_or("").to_string();
                let wind_dir = wind_dir_col.get(i).unwrap_or("").to_string();
                let wind_speed = wind_speed_col.get(i).unwrap_or(0) as f64;
                let wave_height = wave_height_col.get(i).unwrap_or(0) as f64;

                let key = (date as u32, stadium as u8, race as u8);
                weather_map.insert(key, (weather, wind_dir, wind_speed, wave_height));
            }
        }

        Ok(Self {
            weather: weather_map,
        })
    }

    /// Get weather features for a race - O(1)
    pub fn get_weather(
        &self,
        date: u32,
        stadium_code: u8,
        race_no: u8,
    ) -> Option<crate::predictor::WeatherFeatures> {
        self.weather.get(&(date, stadium_code, race_no)).map(
            |(weather, wind_dir, wind_speed, wave_height)| {
                crate::predictor::WeatherFeatures::new(weather, wind_dir, *wind_speed, *wave_height)
            },
        )
    }

    /// Total number of races with weather data
    pub fn len(&self) -> usize {
        self.weather.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.weather.is_empty()
    }
}

/// Stadium-course statistics index for O(1) lookups
pub struct IndexedStadiumCourseStats {
    /// (stadium, course) -> (wins, in2, total)
    stadium_course: HashMap<(u8, u8), (u32, u32, u32)>,
    /// course -> (wins, in2, total) - global stats
    global_course: HashMap<u8, (u32, u32, u32)>,
    /// (racer_id, stadium, course) -> (wins, in2, total)
    racer_stadium_course: HashMap<(u32, u8, u8), (u32, u32, u32)>,
}

impl IndexedStadiumCourseStats {
    /// Global course win rates (fallback)
    const GLOBAL_COURSE_WIN_RATE: [f64; 7] = [0.0, 0.55, 0.14, 0.12, 0.10, 0.06, 0.03];
    const GLOBAL_COURSE_IN2_RATE: [f64; 7] = [0.0, 0.75, 0.35, 0.30, 0.25, 0.20, 0.15];

    /// Build from results CSV (only uses data before cutoff_date to avoid leakage)
    pub fn load<P: AsRef<Path>>(
        csv_path: P,
        cutoff_date: Option<u32>,
    ) -> Result<Self, PolarsError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(csv_path.as_ref().to_path_buf()))?
            .finish()?;

        let date_col = df.column("date")?.i64()?;
        let stadium_col = df.column("stadium_code")?.i64()?;
        let course_col = df.column("course")?.i64()?;
        let rank_col = df.column("rank")?.i64()?;
        let racer_col = df.column("racer_id")?.i64()?;

        let mut stadium_course: HashMap<(u8, u8), (u32, u32, u32)> = HashMap::new();
        let mut global_course: HashMap<u8, (u32, u32, u32)> = HashMap::new();
        let mut racer_stadium_course: HashMap<(u32, u8, u8), (u32, u32, u32)> = HashMap::new();

        for i in 0..df.height() {
            let date = date_col.get(i).unwrap_or(0) as u32;

            // Skip if after cutoff date
            if let Some(cutoff) = cutoff_date {
                if date >= cutoff {
                    continue;
                }
            }

            let stadium = stadium_col.get(i).unwrap_or(0) as u8;
            let course = course_col.get(i).unwrap_or(0) as u8;
            let rank = rank_col.get(i).unwrap_or(0) as u8;
            let racer_id = racer_col.get(i).unwrap_or(0) as u32;

            // Skip invalid data
            if course < 1 || course > 6 || rank < 1 || rank > 6 {
                continue;
            }

            let is_win = if rank == 1 { 1 } else { 0 };
            let is_in2 = if rank <= 2 { 1 } else { 0 };

            // Update stadium-course stats
            let sc_entry = stadium_course.entry((stadium, course)).or_insert((0, 0, 0));
            sc_entry.0 += is_win;
            sc_entry.1 += is_in2;
            sc_entry.2 += 1;

            // Update global course stats
            let gc_entry = global_course.entry(course).or_insert((0, 0, 0));
            gc_entry.0 += is_win;
            gc_entry.1 += is_in2;
            gc_entry.2 += 1;

            // Update racer-stadium-course stats
            let rsc_entry = racer_stadium_course
                .entry((racer_id, stadium, course))
                .or_insert((0, 0, 0));
            rsc_entry.0 += is_win;
            rsc_entry.1 += is_in2;
            rsc_entry.2 += 1;
        }

        Ok(Self {
            stadium_course,
            global_course,
            racer_stadium_course,
        })
    }

    /// Get stadium-course win/in2 rates
    fn get_stadium_course_rates(&self, stadium: u8, course: u8) -> (f64, f64) {
        if let Some(&(wins, in2, total)) = self.stadium_course.get(&(stadium, course)) {
            if total >= 10 {
                return (wins as f64 / total as f64, in2 as f64 / total as f64);
            }
        }
        // Fallback to global
        let idx = course.min(6) as usize;
        (
            Self::GLOBAL_COURSE_WIN_RATE[idx],
            Self::GLOBAL_COURSE_IN2_RATE[idx],
        )
    }

    /// Get global course win rate
    fn get_global_course_win_rate(&self, course: u8) -> f64 {
        if let Some(&(wins, _, total)) = self.global_course.get(&course) {
            if total > 0 {
                return wins as f64 / total as f64;
            }
        }
        Self::GLOBAL_COURSE_WIN_RATE[course.min(6) as usize]
    }

    /// Get racer-stadium-course rates (win_rate, in2_rate) or (0, 0) if not enough data
    fn get_racer_stadium_course_rates(&self, racer_id: u32, stadium: u8, course: u8) -> (f64, f64) {
        if let Some(&(wins, in2, total)) =
            self.racer_stadium_course.get(&(racer_id, stadium, course))
        {
            if total >= 3 {
                return (wins as f64 / total as f64, in2 as f64 / total as f64);
            }
        }
        (0.0, 0.0) // No history
    }

    /// Compute StadiumCourseFeatures for a boat entry
    pub fn compute_features(
        &self,
        stadium: u8,
        boat_no: u8,
        racer_id: u32,
    ) -> crate::data::StadiumCourseFeatures {
        // Use boat_no as expected course
        let course = boat_no;

        let (sc_win, sc_in2) = self.get_stadium_course_rates(stadium, course);
        let global_win = self.get_global_course_win_rate(course);
        let advantage_diff = sc_win - global_win;
        let (racer_win, racer_in2) = self.get_racer_stadium_course_rates(racer_id, stadium, course);

        crate::data::StadiumCourseFeatures {
            stadium_course_win_rate: sc_win,
            stadium_course_in2_rate: sc_in2,
            stadium_course_advantage_diff: advantage_diff,
            racer_course_win_at_stadium: racer_win,
            racer_course_in2_at_stadium: racer_in2,
        }
    }

    /// Number of stadium-course combinations tracked
    pub fn len(&self) -> usize {
        self.stadium_course.len()
    }
}

/// Indexed race info with O(1) lookups for race_type
pub struct IndexedRaceInfo {
    /// race_type indexed by race key
    race_types: HashMap<RaceKey, String>,
}

impl IndexedRaceInfo {
    /// Load and index race info from programs_races.csv
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, PolarsError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(csv_path.as_ref().to_path_buf()))?
            .finish()?;

        let date_col = df.column("date")?.i64()?;
        let stadium_col = df.column("stadium_code")?.i64()?;
        let race_col = df.column("race_no")?.i64()?;
        let race_type_col = df.column("race_type")?.str()?;

        let mut race_types: HashMap<RaceKey, String> = HashMap::new();

        for i in 0..df.height() {
            if let (Some(date), Some(stadium), Some(race)) =
                (date_col.get(i), stadium_col.get(i), race_col.get(i))
            {
                let race_type = race_type_col.get(i).unwrap_or("").to_string();
                let key = (date as u32, stadium as u8, race as u8);
                race_types.insert(key, race_type);
            }
        }

        Ok(Self { race_types })
    }

    /// Get race_type for a race - O(1)
    pub fn get_race_type(&self, date: u32, stadium_code: u8, race_no: u8) -> Option<&str> {
        self.race_types
            .get(&(date, stadium_code, race_no))
            .map(|s| s.as_str())
    }

    /// Encode race_type to (race_grade, is_final)
    /// Matches Python RACE_TYPE_ENCODING iteration order exactly
    pub fn encode_race_context(&self, date: u32, stadium_code: u8, race_no: u8) -> (f64, f64) {
        let race_type = self
            .get_race_type(date, stadium_code, race_no)
            .unwrap_or("");

        // race_grade encoding - must match Python dict iteration order:
        // {"予選": 1, "一般": 1, "特選": 2, "選抜": 2, "準優": 3, "準優勝戦": 3, "優勝戦": 4, "優": 4}
        // Python checks in order and returns first match
        let race_grade = if race_type.contains("予選") {
            1.0
        } else if race_type.contains("一般") {
            1.0
        } else if race_type.contains("特選") {
            2.0
        } else if race_type.contains("選抜") {
            2.0
        } else if race_type.contains("準優") {
            3.0 // Matches both "準優" and "準優勝戦"
        } else if race_type.contains("優勝戦") {
            4.0
        } else if race_type.contains("優") {
            4.0
        } else {
            1.0 // Default
        };

        // is_final: 1.0 if race_grade >= 3 (準優 or 優勝)
        let is_final = if race_grade >= 3.0 { 1.0 } else { 0.0 };

        (race_grade, is_final)
    }

    /// Total number of races
    pub fn len(&self) -> usize {
        self.race_types.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.race_types.is_empty()
    }
}

/// Bundle of indexed feature sources for computing the full feature set of a
/// single race outside the backtest loop.
///
/// Used by the `predict` and `today` commands so their predictions match the
/// backtest's. Historical features come from `results_entries.csv` (required);
/// exhibition times, weather, and race context are best-effort and degrade to
/// model defaults when their source files are absent (e.g. for future races).
pub struct RaceFeatureContext {
    history: crate::data::RacerHistoryIndex,
    results: Option<IndexedResultsData>,
    weather: Option<IndexedWeatherData>,
    race_info: Option<IndexedRaceInfo>,
}

impl RaceFeatureContext {
    /// Load all available feature sources from a processed-data directory.
    ///
    /// History (from `results_entries.csv`) is required; the remaining sources
    /// are optional and silently skipped when missing.
    pub fn load(data_dir: &Path) -> Result<Self, PolarsError> {
        let results_path = data_dir.join("results_entries.csv");
        let history = crate::data::RacerHistoryIndex::load(&results_path)?;
        let results = IndexedResultsData::load(&results_path).ok();

        let weather_path = data_dir.join("results_races.csv");
        let weather = if weather_path.exists() {
            IndexedWeatherData::load(&weather_path).ok()
        } else {
            None
        };

        let race_info_path = data_dir.join("programs_races.csv");
        let race_info = if race_info_path.exists() {
            IndexedRaceInfo::load(&race_info_path).ok()
        } else {
            None
        };

        Ok(Self {
            history,
            results,
            weather,
            race_info,
        })
    }

    /// Compute historical features for each entry, considering only races
    /// before `date`. Returned vector is parallel to `entries`.
    pub fn historical_features(
        &self,
        date: u32,
        stadium_code: u8,
        entries: &[RacerEntry],
    ) -> Vec<crate::data::HistoricalFeatures> {
        entries
            .iter()
            .map(|e| {
                self.history.compute_historical_features(
                    e.racer_id,
                    stadium_code,
                    e.boat_no, // course (assuming boat_no = starting course)
                    e.boat_no,
                    date,
                )
            })
            .collect()
    }

    /// Exhibition times for the race, if recorded in results data.
    pub fn exhibition_times(&self, date: u32, stadium_code: u8, race_no: u8) -> Option<[f64; 6]> {
        self.results
            .as_ref()
            .and_then(|r| r.get_exhibition_times(date, stadium_code, race_no))
    }

    /// Weather features for the race, if recorded.
    pub fn weather(
        &self,
        date: u32,
        stadium_code: u8,
        race_no: u8,
    ) -> Option<crate::predictor::WeatherFeatures> {
        self.weather
            .as_ref()
            .and_then(|w| w.get_weather(date, stadium_code, race_no))
    }

    /// Encoded race context (race_grade, is_final), if race info is available.
    pub fn race_context(&self, date: u32, stadium_code: u8, race_no: u8) -> Option<(f64, f64)> {
        self.race_info
            .as_ref()
            .map(|ri| ri.encode_race_context(date, stadium_code, race_no))
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
    /// Bet on highest probability combinations (instead of highest EV)
    pub bet_by_probability: bool,
    /// Maximum odds to bet on (filters out longshots where model is unreliable)
    /// Default: None (no limit for synthetic odds), 30.0 recommended for real odds
    pub max_odds: Option<f64>,
    /// Decide bets from the latest odds snapshot captured strictly before each
    /// race's deadline (fail-closed), instead of the post-close settled odds.
    pub lookahead_free: bool,
    /// Directory of accumulated `OddsSnapshot` files for lookahead-free mode.
    /// Falls back to `odds_dir` when None.
    pub snapshot_dir: Option<PathBuf>,
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
            bet_by_probability: false,
            max_odds: None,
            lookahead_free: false,
            snapshot_dir: None,
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

        // Load the predictor from the configured model directory (detects model
        // type from metadata.json; fails soft to the heuristic fallback).
        let predictor = UnifiedPredictor::from_model_dir(config.model_dir.as_deref());

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

        // Load race info for race_type
        let races_path = programs_path
            .as_ref()
            .parent()
            .map(|p| p.join("programs_races.csv"))
            .unwrap_or_else(|| PathBuf::from("programs_races.csv"));
        let race_info = if races_path.exists() {
            eprintln!("Loading race info...");
            match IndexedRaceInfo::load(&races_path) {
                Ok(info) => {
                    eprintln!("Loaded {} race info entries", info.len());
                    Some(info)
                }
                Err(e) => {
                    eprintln!("Failed to load race info: {}, using defaults", e);
                    None
                }
            }
        } else {
            eprintln!("Race info file not found, using defaults");
            None
        };

        // Load weather data from results_races.csv
        let results_races_path = results_path
            .as_ref()
            .parent()
            .map(|p| p.join("results_races.csv"))
            .unwrap_or_else(|| PathBuf::from("results_races.csv"));
        let weather_data = if results_races_path.exists() {
            eprintln!("Loading weather data...");
            match IndexedWeatherData::load(&results_races_path) {
                Ok(data) => {
                    eprintln!("Loaded weather for {} races", data.len());
                    Some(data)
                }
                Err(e) => {
                    eprintln!("Failed to load weather data: {}, using defaults", e);
                    None
                }
            }
        } else {
            eprintln!("Weather data file not found, using defaults");
            None
        };

        // Load racer history index for historical features
        eprintln!("Loading racer history...");
        let history_index = crate::data::RacerHistoryIndex::load(&results_path)?;
        eprintln!("Loaded history for {} racers", history_index.len());

        // Note: Stadium course features are disabled for 50-feature models
        // To enable, uncomment below and pass stadium_course_features to predictor
        // eprintln!("Loading stadium course stats...");
        // let stadium_course_stats = IndexedStadiumCourseStats::load(&results_path, self.config.test_start_date)?;
        // eprintln!("Loaded {} stadium-course combinations", stadium_course_stats.len());

        // Lookahead-free mode: index all snapshots once (avoids re-reading the
        // directory per race). Built empty when disabled, so it costs nothing.
        let snapshot_index = if self.config.lookahead_free {
            let dir = self
                .config
                .snapshot_dir
                .clone()
                .or_else(|| odds_dir.map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("data/odds"));
            eprintln!(
                "Lookahead-free mode: indexing pre-deadline snapshots in {}...",
                dir.display()
            );
            SnapshotIndex::load(&dir)
        } else {
            SnapshotIndex::load(PathBuf::new())
        };

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

            // Load odds. Lookahead-free mode selects the latest snapshot captured
            // strictly before the deadline; the default path uses the post-close
            // settled file (the documented, biased baseline).
            let mut used_synthetic = false;
            let odds = if self.config.lookahead_free {
                match snapshot_index.select_exacta(*date, *stadium_code, *race_no) {
                    PreDeadlineOdds::Selected { odds, .. } => odds,
                    // Fail-closed: never fall through to post-close odds. Use
                    // synthetic if enabled, else exclude this race from betting.
                    PreDeadlineOdds::OnlyPostClose | PreDeadlineOdds::NoSnapshots => {
                        if let Some(ref synth) = self.synthetic_odds {
                            result.races_fell_back_to_synthetic += 1;
                            used_synthetic = true;
                            synth.get_all_odds()
                        } else {
                            result.races_excluded_no_predeadline_odds += 1;
                            continue;
                        }
                    }
                }
            } else {
                // Load odds if available, or use synthetic odds
                let real_odds: Option<HashMap<(u8, u8), f64>> =
                    odds_dir.and_then(|dir| load_exacta_odds(dir, *date, *stadium_code, *race_no));

                match real_odds {
                    Some(o) => o,
                    None => {
                        // Use synthetic odds if enabled, otherwise skip
                        if let Some(ref synth) = self.synthetic_odds {
                            synth.get_all_odds()
                        } else {
                            continue;
                        }
                    }
                }
            };

            // Compute historical features for each racer
            let historical_features: Vec<_> = entries
                .iter()
                .map(|e| {
                    history_index.compute_historical_features(
                        e.racer_id,
                        *stadium_code,
                        e.boat_no, // course (assuming boat_no = starting course)
                        e.boat_no, // boat_no for course-taking features
                        *date,     // only consider races before this date
                    )
                })
                .collect();

            // Get exhibition times from results data
            let exhibition_times =
                results_data.get_exhibition_times(*date, *stadium_code, *race_no);

            // Get race context (race_grade, is_final) from race info.
            // Only use real race context with real odds; synthetic odds don't
            // reflect race context, causing EV mismatch. In lookahead-free mode
            // the choice is per-race (a race may use real odds while another
            // fell back to synthetic), so it keys off `used_synthetic`.
            let real_context = || {
                race_info
                    .as_ref()
                    .map(|info| info.encode_race_context(*date, *stadium_code, *race_no))
            };
            let race_context = if self.config.lookahead_free {
                if used_synthetic {
                    None
                } else {
                    real_context()
                }
            } else if self.synthetic_odds.is_none() {
                real_context()
            } else {
                None // Use defaults (1.0, 0.0) for synthetic odds
            };

            // Run prediction with historical features, exhibition times, race context, stadium, and weather
            let weather = weather_data
                .as_ref()
                .and_then(|wd| wd.get_weather(*date, *stadium_code, *race_no));
            let racer_entries: Vec<_> = entries.iter().map(|e| e.to_racer_entry()).collect();

            // Note: stadium_course is None for 50-feature models (disabled)
            let position_probs = self.predictor.predict_positions_full(
                &racer_entries,
                Some(&historical_features),
                None, // stadium_course disabled for 50-feature model
                exhibition_times,
                race_context,
                Some(*stadium_code),
                weather.as_ref(),
            );
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
                .filter(|(_, _, prob, odds, ev)| {
                    // Apply max_odds filter if configured
                    let max_odds = self.config.max_odds.unwrap_or(f64::MAX);
                    if *odds > max_odds {
                        return false;
                    }

                    if self.config.bet_by_probability {
                        // When betting by probability, skip extreme longshots
                        // Model is unreliable for very low probabilities
                        *prob >= 0.01
                    } else {
                        // EV strategy: require EV > threshold
                        *ev > self.config.ev_threshold
                    }
                })
                .collect();

            if value_bets.is_empty() {
                continue;
            }

            // Sort by probability (if by_prob) or EV (default) descending
            if self.config.bet_by_probability {
                value_bets
                    .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                value_bets
                    .sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
            }

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
        if self.config.bet_by_probability {
            println!("Strategy: bet by PROBABILITY (top-N)");
        } else {
            println!("Strategy: bet by EV (EV > threshold)");
        }
        if let Some(max_odds) = self.config.max_odds {
            println!("Max odds: {:.1}", max_odds);
        }
        println!("{}", "-".repeat(60));
        println!("Total races: {}", result.total_races);
        println!("Races with bets: {}", result.races_with_bets);
        if self.config.lookahead_free {
            let total = result.total_races.max(1);
            println!(
                "Races excluded (no pre-deadline odds): {} ({:.1}%)",
                result.races_excluded_no_predeadline_odds,
                result.races_excluded_no_predeadline_odds as f64 / total as f64 * 100.0
            );
            println!(
                "Races fell back to synthetic: {}",
                result.races_fell_back_to_synthetic
            );
        }
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

        // Bet distribution analysis
        if !result.bets.is_empty() {
            println!("{}", "-".repeat(60));
            println!("BET DISTRIBUTION:");

            // Group by odds ranges
            let mut odds_bins: [(usize, usize, i64, i64); 5] = [(0, 0, 0, 0); 5]; // count, wins, stake, profit
            for bet in &result.bets {
                let bin = if bet.odds < 10.0 {
                    0
                } else if bet.odds < 30.0 {
                    1
                } else if bet.odds < 100.0 {
                    2
                } else if bet.odds < 300.0 {
                    3
                } else {
                    4
                };
                odds_bins[bin].0 += 1;
                if bet.won {
                    odds_bins[bin].1 += 1;
                }
                odds_bins[bin].2 += bet.stake;
                odds_bins[bin].3 += bet.profit;
            }

            let labels = ["<10", "10-30", "30-100", "100-300", ">300"];
            for (i, label) in labels.iter().enumerate() {
                let (count, wins, stake, profit) = odds_bins[i];
                if count > 0 {
                    let roi = if stake > 0 {
                        profit as f64 / stake as f64 * 100.0
                    } else {
                        0.0
                    };
                    println!(
                        "  Odds {}: {} bets, {} wins ({:.1}%), ROI: {:+.1}%",
                        label,
                        count,
                        wins,
                        wins as f64 / count as f64 * 100.0,
                        roi
                    );
                }
            }

            // Average probability and odds
            let avg_prob: f64 =
                result.bets.iter().map(|b| b.probability).sum::<f64>() / result.bets.len() as f64;
            let avg_odds: f64 =
                result.bets.iter().map(|b| b.odds).sum::<f64>() / result.bets.len() as f64;
            println!("\n  Avg probability: {:.2}%", avg_prob * 100.0);
            println!("  Avg odds: {:.1}", avg_odds);
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
