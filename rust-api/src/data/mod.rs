//! Data loading modules for CSV and JSON files

pub mod csv_loader;
pub mod odds_loader;

// Re-export commonly used types
pub use csv_loader::{ProgramEntry, RaceData};
pub use odds_loader::{load_exacta_odds, load_trifecta_odds};
