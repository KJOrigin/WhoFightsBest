set -e
set -u

MAD_PATH=/your/path

python3 $MAD_PATH/code/mbti_debate.py \
    -i $MAD_PATH/your/data/path \
    -o $MAD_PATH/your/output/path \
    -m MODEL_NAME \
    -k " " \
    --datasets "chemistry,computer science,biology,business,economics,health,history,law,engineering,math,other,philosophy,physics,psychology"