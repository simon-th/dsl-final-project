source .env

LEFT_AGENT="${LEFT_AGENT:-"ppo"}"
LEFT_PATH="${LEFT_PATH:-""}"
RIGHT_AGENT="${RIGHT_AGENT:-"cma"}"
RIGHT_PATH="${RIGHT_PATH:-""}"

SLIMEVOLLEYGYM_PATH="${SLIMEVOLLEYGYM_PATH:-"./slimevolleygym"}"
RENDER=${RENDER:-false}

if [ "$RENDER" = true ]; then
    python ${SLIMEVOLLEYGYM_PATH}/eval_agents.py --left ${LEFT_AGENT} --leftpath ${LEFT_PATH} --right ${RIGHT_AGENT} --rightpath ${RIGHT_PATH} --render
else
    python ${SLIMEVOLLEYGYM_PATH}/eval_agents.py --left ${LEFT_AGENT} --leftpath ${LEFT_PATH} --right ${RIGHT_AGENT} --rightpath ${RIGHT_PATH}
fi
