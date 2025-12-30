//! Backtesting engine for validating betting strategies

pub mod metrics;
pub mod simulator;

pub use metrics::{calculate_metrics, BacktestMetrics};
pub use simulator::{BacktestConfig, BacktestResult, BacktestSimulator, BetRecord};
