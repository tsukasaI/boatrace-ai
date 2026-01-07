//! Backtesting engine for validating betting strategies

pub mod metrics;
pub mod output;
pub mod simulator;
pub mod synthetic;

pub use metrics::{calculate_metrics, BacktestMetrics};
pub use output::{BacktestOutput, OutputFormat};
pub use simulator::{BacktestConfig, BacktestResult, BacktestSimulator, BetRecord};
pub use synthetic::SyntheticOddsGenerator;
