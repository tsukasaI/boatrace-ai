use actix_web::{http::StatusCode, HttpResponse, ResponseError};
use std::fmt;

use crate::models::ErrorResponse;

/// Application error types
#[derive(Debug)]
pub enum AppError {
    /// Invalid request data
    ValidationError(String),
    /// Model or prediction error
    PredictionError(String),
    /// Internal server error
    InternalError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AppError::PredictionError(msg) => write!(f, "Prediction error: {}", msg),
            AppError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for AppError {}

impl ResponseError for AppError {
    fn status_code(&self) -> StatusCode {
        match self {
            AppError::ValidationError(_) => StatusCode::BAD_REQUEST,
            AppError::PredictionError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_response(&self) -> HttpResponse {
        let (error_code, message) = match self {
            AppError::ValidationError(msg) => ("validation_error", msg.clone()),
            AppError::PredictionError(msg) => ("prediction_error", msg.clone()),
            AppError::InternalError(msg) => ("internal_error", msg.clone()),
        };

        HttpResponse::build(self.status_code()).json(ErrorResponse {
            error: error_code.to_string(),
            message,
        })
    }
}

/// Validation functions
pub fn validate_entries_count(count: usize) -> Result<(), AppError> {
    if count != 6 {
        return Err(AppError::ValidationError(format!(
            "Exactly 6 racer entries required, got {}",
            count
        )));
    }
    Ok(())
}

pub fn validate_boat_number(boat_no: u8) -> Result<(), AppError> {
    if !(1..=6).contains(&boat_no) {
        return Err(AppError::ValidationError(format!(
            "Boat number must be between 1 and 6, got {}",
            boat_no
        )));
    }
    Ok(())
}

pub fn validate_odds(odds: f64) -> Result<(), AppError> {
    if odds < 0.0 {
        return Err(AppError::ValidationError(format!(
            "Odds must be non-negative, got {}",
            odds
        )));
    }
    Ok(())
}

pub fn validate_probability(prob: f64) -> Result<(), AppError> {
    if !(0.0..=1.0).contains(&prob) {
        return Err(AppError::ValidationError(format!(
            "Probability must be between 0 and 1, got {}",
            prob
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_entries_count_valid() {
        assert!(validate_entries_count(6).is_ok());
    }

    #[test]
    fn test_validate_entries_count_invalid() {
        assert!(validate_entries_count(5).is_err());
        assert!(validate_entries_count(7).is_err());
        assert!(validate_entries_count(0).is_err());
    }

    #[test]
    fn test_validate_boat_number_valid() {
        for i in 1..=6 {
            assert!(validate_boat_number(i).is_ok());
        }
    }

    #[test]
    fn test_validate_boat_number_invalid() {
        assert!(validate_boat_number(0).is_err());
        assert!(validate_boat_number(7).is_err());
    }

    #[test]
    fn test_validate_odds_valid() {
        assert!(validate_odds(0.0).is_ok());
        assert!(validate_odds(5.5).is_ok());
        assert!(validate_odds(100.0).is_ok());
    }

    #[test]
    fn test_validate_odds_invalid() {
        assert!(validate_odds(-1.0).is_err());
    }

    #[test]
    fn test_validate_probability_valid() {
        assert!(validate_probability(0.0).is_ok());
        assert!(validate_probability(0.5).is_ok());
        assert!(validate_probability(1.0).is_ok());
    }

    #[test]
    fn test_validate_probability_invalid() {
        assert!(validate_probability(-0.1).is_err());
        assert!(validate_probability(1.1).is_err());
    }

    #[test]
    fn test_error_display() {
        let err = AppError::ValidationError("test error".to_string());
        assert!(err.to_string().contains("Validation error"));
    }

    #[test]
    fn test_error_status_codes() {
        assert_eq!(
            AppError::ValidationError("".to_string()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            AppError::PredictionError("".to_string()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            AppError::InternalError("".to_string()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
