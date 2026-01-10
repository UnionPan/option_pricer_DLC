"""
API endpoints for Deep RL Hedging.
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.rl_hedging import (
    RLInferenceRequest,
    RLInferenceResponse,
)
from backend.services.rl_hedging_service import RLHedgingService

router = APIRouter()


@router.post("/inference", response_model=RLInferenceResponse)
async def run_rl_inference(request: RLInferenceRequest):
    """
    Run inference with a pre-trained RL hedging agent.

    This endpoint demonstrates various RL agents on a configurable
    option hedging environment:
    - PPO (Proximal Policy Optimization)
    - SAC (Soft Actor-Critic)
    - TD3 (Twin Delayed DDPG)
    - Deep Hedging Networks
    - AIS (Adaptive Importance Sampling) Hedging

    Args:
        request: Configuration including agent type and environment parameters

    Returns:
        Agent performance including hedging trajectory, P&L, and metrics
    """
    try:
        result = RLHedgingService.run_inference(
            agent_type=request.agent_type,
            environment_config=request.environment_config.dict(),
            use_demo_model=request.use_demo_model,
            model_path=request.model_path,
            random_seed=request.random_seed,
        )

        return RLInferenceResponse(**result)

    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL inference failed: {str(e)}")


@router.get("/agents")
async def list_available_agents():
    """
    List all available pre-trained RL agents.

    Returns:
        List of agent metadata including type, description, and capabilities
    """
    agents = [
        {
            "type": "ppo",
            "name": "PPO Agent",
            "description": "Proximal Policy Optimization - stable on-policy RL",
            "features": [
                "Clipped objective for stable training",
                "Handles continuous action spaces",
                "Good for high-dimensional state spaces",
            ],
            "available": True,
        },
        {
            "type": "sac",
            "name": "SAC Agent",
            "description": "Soft Actor-Critic - maximum entropy off-policy RL",
            "features": [
                "Automatic temperature tuning",
                "Sample-efficient off-policy learning",
                "Robust to hyperparameter choices",
            ],
            "available": True,
        },
        {
            "type": "td3",
            "name": "TD3 Agent",
            "description": "Twin Delayed DDPG - robust deterministic policy",
            "features": [
                "Reduced overestimation bias",
                "Delayed policy updates",
                "Target policy smoothing",
            ],
            "available": True,
        },
        {
            "type": "deep_hedging",
            "name": "Deep Hedging Network",
            "description": "End-to-end neural network for hedging optimization",
            "features": [
                "Direct optimization of hedging P&L",
                "Handles transaction costs natively",
                "No-arbitrage constraints via convex layers",
            ],
            "available": True,
        },
        {
            "type": "ais_hedging",
            "name": "AIS Hedging Agent",
            "description": "Adaptive Importance Sampling for tail risk hedging",
            "features": [
                "Efficient rare event simulation",
                "CVaR-focused hedging strategies",
                "Cross-entropy method optimization",
            ],
            "available": True,
        },
    ]

    return {"agents": agents}
