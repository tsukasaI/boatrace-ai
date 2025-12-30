use actix_web::{HttpResponse, Responder};
use crate::models::HealthResponse;

/// Health check endpoint
pub async fn health_check() -> impl Responder {
    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_loaded: false, // TODO: Check actual model status
    };

    HttpResponse::Ok().json(response)
}
