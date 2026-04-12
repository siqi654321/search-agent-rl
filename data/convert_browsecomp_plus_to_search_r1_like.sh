#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
HF_REPO_ID="${HF_REPO_ID:-Tevatron/browsecomp-plus}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/browsecompplus_searchr1}"

python3 "$SCRIPT_DIR/convert_browsecomp_plus_to_search_r1_like.py" \
--hf_repo_id "$HF_REPO_ID" \
--output_dir "$OUTPUT_DIR" \
--data_source searchR1_openresearcher \
--system_prompt "You are a helpful AI assistant that can search for information to answer questions accurately.\n\nWhen answering questions:\n1. Use the available search tools to find relevant and reliable information\n2. Synthesize information from multiple sources when needed\n3. Provide accurate and comprehensive answers based on your search results\n4. Do not use Chinese in your responses, keep using English only\n5. Do not search the same query multiple times\n6. Do not call tools inside <think></think>\n7. Always put your final answer in \\<answer></answer> format\n\nFor example:\n- If the answer is "American", write: <answer>American</answer>\n- If the answer is "yes", write: <answer>yes</answer>\n- If the answer is a year like "1985", write: <answer>1985</answer>\n\nRemember to search thoroughly and provide your final answer clearly within the <answer></answer> format.\n\n" \
--reward_metric token_f1 \
--user_prompt_prefix "" \
--test_ratio 0.0 \
# --dry_run 
