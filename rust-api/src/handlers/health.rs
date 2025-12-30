use actix_web::{web, HttpResponse, Responder};
use std::sync::Arc;

use crate::models::HealthResponse;
use crate::AppState;

/// Health check endpoint
pub async fn health_check(state: web::Data<Arc<AppState>>) -> impl Responder {
    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_loaded: state.predictor.is_some(),
    };

    HttpResponse::Ok().json(response)
}
