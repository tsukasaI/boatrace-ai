//! Boatrace AI - Boat racing prediction system
//!
//! This library provides:
//! - Position probability prediction using ONNX models
//! - Exacta (2連単) and Trifecta (3連単) probability calculation
//! - Kelly criterion bet sizing
//! - Data loading and processing utilities
//!
//! # Example
//!
//! ```no_run
//! use boatrace::core::kelly::KellyCalculator;
//! use boatrace::predictor::FallbackPredictor;
//!
//! // Create a predictor
//! let predictor = FallbackPredictor::new();
//!
//! // Create a Kelly calculator
//! let calc = KellyCalculator::with_defaults(100_000);
//! let sizing = calc.calculate_single(0.25, 5.0);
//! println!("Recommended stake: {}", sizing.stake);
//! ```

pub mod core;
pub mod data;
pub mod models;
pub mod predictor;

// API-specific modules (only available with api feature)
#[cfg(feature = "api")]
pub mod error;

// Re-export commonly used types
pub use data::{load_exacta_odds, load_trifecta_odds, ProgramEntry, RaceData};
pub use models::{
    ExactaOdds, ExactaPrediction, PositionProb, PredictRequest, PredictResponse, RacerEntry,
    TrifectaOdds, TrifectaPrediction,
};
pub use predictor::{FallbackPredictor, Predictor};
