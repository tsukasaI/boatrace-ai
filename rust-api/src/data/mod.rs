//! Data loading and feature engineering modules

pub mod csv_loader;
pub mod features;
pub mod odds_loader;

// Re-export commonly used types
pub use csv_loader::{ProgramEntry, RaceData};
pub use features::{
    get_all_feature_names, get_base_feature_names, BaseFeatures, FeatureEngineering,
    HistoricalFeatures, RacerFeatures, RelativeFeatures,
};
pub use odds_loader::{load_exacta_odds, load_trifecta_odds};
