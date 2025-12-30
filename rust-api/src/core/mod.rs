//! Core business logic modules

pub mod kelly;

// Re-export commonly used types
pub use kelly::{calculate_kelly_fraction, calculate_optimal_stake, BetSizing, KellyCalculator};
