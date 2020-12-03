source .env

LEFT_AGENT="${LEFT_AGENT:-"ppo"}"
RIGHT_AGENT="${RIGHT_AGENT:-"cma"}"
SLIMEVOLLEYGYM_PATH="${SLIMEVOLLEYGYM_PATH:-"./slimevolleygym"}"
RENDER=${RENDER:-false}

if [ "$RENDER" = true ]; then
    python ${SLIMEVOLLEYGYM_PATH}/eval_agents.py --left ${LEFT_AGENT} --right ${RIGHT_AGENT} --render
else
    python ${SLIMEVOLLEYGYM_PATH}/eval_agents.py --left ${LEFT_AGENT} --right ${RIGHT_AGENT}
fi
