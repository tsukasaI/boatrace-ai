//! Output formatting for backtest results
//!
//! Supports JSON, CSV, and table (terminal) output formats.

use super::metrics::{
    analyze_by_odds_range, analyze_by_stadium, BacktestMetrics, DimensionAnalysis,
};
use super::simulator::{BacktestConfig, BacktestResult, BetRecord};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::str::FromStr;

/// Output format for backtest results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Terminal table output (default)
    Table,
    /// JSON format
    Json,
    /// CSV format
    Csv,
}

impl FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "table" => Ok(OutputFormat::Table),
            "json" => Ok(OutputFormat::Json),
            "csv" => Ok(OutputFormat::Csv),
            _ => Err(format!(
                "Unknown output format: '{}'. Use 'table', 'json', or 'csv'",
                s
            )),
        }
    }
}

/// Configuration snapshot for output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfigOutput {
    pub ev_threshold: f64,
    pub stake: i64,
    pub max_bets_per_race: usize,
    pub bet_by_probability: bool,
    pub max_odds: Option<f64>,
    pub test_start_date: Option<u32>,
    pub use_synthetic_odds: bool,
}

impl From<&BacktestConfig> for BacktestConfigOutput {
    fn from(config: &BacktestConfig) -> Self {
        Self {
            ev_threshold: config.ev_threshold,
            stake: config.stake,
            max_bets_per_race: config.max_bets_per_race,
            bet_by_probability: config.bet_by_probability,
            max_odds: config.max_odds,
            test_start_date: config.test_start_date,
            use_synthetic_odds: config.use_synthetic_odds,
        }
    }
}

/// Summary statistics for output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestSummaryOutput {
    pub total_races: usize,
    pub races_with_bets: usize,
    pub total_bets: usize,
    pub winning_bets: usize,
    pub total_stake: i64,
    pub total_payout: i64,
    pub total_profit: i64,
    pub roi: f64,
}

impl From<&BacktestResult> for BacktestSummaryOutput {
    fn from(result: &BacktestResult) -> Self {
        let winning_bets = result.bets.iter().filter(|b| b.won).count();
        Self {
            total_races: result.total_races,
            races_with_bets: result.races_with_bets,
            total_bets: result.bets.len(),
            winning_bets,
            total_stake: result.total_stake,
            total_payout: result.total_payout,
            total_profit: result.total_profit(),
            roi: result.roi(),
        }
    }
}

/// Dimension analysis output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisOutput {
    pub by_stadium: Vec<DimensionAnalysis>,
    pub by_odds_range: Vec<DimensionAnalysis>,
}

/// Complete backtest output for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestOutput {
    pub config: BacktestConfigOutput,
    pub summary: BacktestSummaryOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<BacktestMetrics>,
    pub analysis: AnalysisOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bets: Option<Vec<BetRecord>>,
}

impl BacktestOutput {
    /// Create output from backtest result and config
    pub fn from_result(
        result: &BacktestResult,
        config: &BacktestConfig,
        include_bets: bool,
    ) -> Self {
        Self {
            config: BacktestConfigOutput::from(config),
            summary: BacktestSummaryOutput::from(result),
            metrics: result.metrics.clone(),
            analysis: AnalysisOutput {
                by_stadium: analyze_by_stadium(&result.bets),
                by_odds_range: analyze_by_odds_range(&result.bets),
            },
            bets: if include_bets {
                Some(result.bets.clone())
            } else {
                None
            },
        }
    }

    /// Write as JSON to writer
    pub fn write_json<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(writer, "{}", json)
    }

    /// Write summary CSV to writer (single row with key metrics)
    pub fn write_summary_csv<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let mut wtr = csv::Writer::from_writer(writer);

        // Write header
        wtr.write_record([
            "total_races",
            "races_with_bets",
            "total_bets",
            "winning_bets",
            "hit_rate",
            "total_stake",
            "total_payout",
            "total_profit",
            "roi",
            "avg_ev",
            "avg_odds",
            "avg_probability",
            "profit_factor",
            "max_drawdown",
        ])
        .map_err(csv_to_io_error)?;

        // Get metrics or use defaults
        let metrics = self.metrics.as_ref();

        wtr.write_record([
            self.summary.total_races.to_string(),
            self.summary.races_with_bets.to_string(),
            self.summary.total_bets.to_string(),
            self.summary.winning_bets.to_string(),
            format!("{:.4}", metrics.map(|m| m.hit_rate).unwrap_or(0.0)),
            self.summary.total_stake.to_string(),
            self.summary.total_payout.to_string(),
            self.summary.total_profit.to_string(),
            format!("{:.4}", self.summary.roi),
            format!("{:.4}", metrics.map(|m| m.avg_ev).unwrap_or(0.0)),
            format!("{:.2}", metrics.map(|m| m.avg_odds).unwrap_or(0.0)),
            format!("{:.4}", metrics.map(|m| m.avg_probability).unwrap_or(0.0)),
            format!("{:.4}", metrics.map(|m| m.profit_factor).unwrap_or(0.0)),
            metrics.map(|m| m.max_drawdown).unwrap_or(0).to_string(),
        ])
        .map_err(csv_to_io_error)?;

        wtr.flush()
    }

    /// Write detailed bet records to CSV
    pub fn write_bets_csv<W: Write>(bets: &[BetRecord], writer: &mut W) -> std::io::Result<()> {
        let mut wtr = csv::Writer::from_writer(writer);

        // Write header
        wtr.write_record([
            "date",
            "stadium_code",
            "race_no",
            "combination",
            "probability",
            "odds",
            "expected_value",
            "stake",
            "actual_first",
            "actual_second",
            "won",
            "profit",
        ])
        .map_err(csv_to_io_error)?;

        // Write bet records
        for bet in bets {
            wtr.write_record([
                bet.date.to_string(),
                bet.stadium_code.to_string(),
                bet.race_no.to_string(),
                format!("{}-{}", bet.first, bet.second),
                format!("{:.6}", bet.probability),
                format!("{:.1}", bet.odds),
                format!("{:.4}", bet.expected_value),
                bet.stake.to_string(),
                bet.actual_first.to_string(),
                bet.actual_second.to_string(),
                bet.won.to_string(),
                bet.profit.to_string(),
            ])
            .map_err(csv_to_io_error)?;
        }

        wtr.flush()
    }
}

/// Convert csv::Error to std::io::Error
fn csv_to_io_error(e: csv::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_result() -> BacktestResult {
        let mut result = BacktestResult::new();
        result.total_races = 100;
        result.races_with_bets = 50;
        result.total_stake = 5000;
        result.total_payout = 6000;
        result.bets = vec![
            BetRecord {
                date: 20240715,
                stadium_code: 23,
                race_no: 5,
                first: 1,
                second: 3,
                probability: 0.08,
                odds: 12.5,
                expected_value: 1.0,
                stake: 100,
                actual_first: 1,
                actual_second: 3,
                won: true,
                profit: 1150,
            },
            BetRecord {
                date: 20240715,
                stadium_code: 23,
                race_no: 6,
                first: 2,
                second: 1,
                probability: 0.05,
                odds: 25.0,
                expected_value: 1.25,
                stake: 100,
                actual_first: 1,
                actual_second: 2,
                won: false,
                profit: -100,
            },
        ];
        result.finalize();
        result
    }

    fn create_test_config() -> BacktestConfig {
        BacktestConfig {
            ev_threshold: 1.0,
            stake: 100,
            max_bets_per_race: 3,
            use_kelly: false,
            kelly_multiplier: 0.25,
            test_start_date: Some(20240701),
            model_dir: None,
            use_synthetic_odds: false,
            bet_by_probability: false,
            max_odds: None,
        }
    }

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(
            OutputFormat::from_str("table").unwrap(),
            OutputFormat::Table
        );
        assert_eq!(OutputFormat::from_str("JSON").unwrap(), OutputFormat::Json);
        assert_eq!(OutputFormat::from_str("csv").unwrap(), OutputFormat::Csv);
        assert!(OutputFormat::from_str("invalid").is_err());
    }

    #[test]
    fn test_json_output() {
        let result = create_test_result();
        let config = create_test_config();
        let output = BacktestOutput::from_result(&result, &config, false);

        let mut buf = Vec::new();
        output.write_json(&mut buf).unwrap();

        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(json["summary"]["total_races"], 100);
        assert_eq!(json["summary"]["total_bets"], 2);
        assert!(json["bets"].is_null());
    }

    #[test]
    fn test_json_output_with_bets() {
        let result = create_test_result();
        let config = create_test_config();
        let output = BacktestOutput::from_result(&result, &config, true);

        let mut buf = Vec::new();
        output.write_json(&mut buf).unwrap();

        let json: serde_json::Value = serde_json::from_slice(&buf).unwrap();
        assert!(json["bets"].is_array());
        assert_eq!(json["bets"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_summary_csv_output() {
        let result = create_test_result();
        let config = create_test_config();
        let output = BacktestOutput::from_result(&result, &config, false);

        let mut buf = Vec::new();
        output.write_summary_csv(&mut buf).unwrap();

        let csv_str = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = csv_str.lines().collect();
        assert_eq!(lines.len(), 2); // header + 1 data row
        assert!(lines[0].contains("total_races"));
        assert!(lines[1].contains("100")); // total_races
    }

    #[test]
    fn test_bets_csv_output() {
        let result = create_test_result();

        let mut buf = Vec::new();
        BacktestOutput::write_bets_csv(&result.bets, &mut buf).unwrap();

        let csv_str = String::from_utf8(buf).unwrap();
        let lines: Vec<&str> = csv_str.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 bets
        assert!(lines[0].contains("date"));
        assert!(lines[1].contains("1-3")); // combination
    }
}
