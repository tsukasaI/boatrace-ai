//! Data loading and feature engineering modules

pub mod csv_loader;
pub mod features;
pub mod odds_loader;
pub mod parser;

// Re-export commonly used types
pub use csv_loader::{IndexedRaceData, ProgramEntry, RaceData, RaceKey};
pub use features::{
    get_all_feature_names, get_base_feature_names, BaseFeatures, FeatureEngineering,
    HistoricalFeatures, RacerFeatures, RelativeFeatures,
};
pub use odds_loader::{load_exacta_odds, load_trifecta_odds};
pub use parser::{
    flatten_payouts, ParsedRacerEntry, PayoutParser, PayoutRecord, ProgramParser, RaceInfo,
    RacePayouts, RaceResult, ResultParser,
};
