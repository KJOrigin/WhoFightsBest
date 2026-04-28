import os
import json
import random
import re
import argparse
from utils.mbti_url import Agent
from datetime import datetime
from tqdm import tqdm
import time
import glob

def load_mbti_data(mbti_path):
    try:
        with open(mbti_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading MBTI data: {e}")
        return {}

NAME_LIST = [
    "Proponent",    
    "Opponent",     
    "Moderator",   
]

MBTI_TYPES = [
    "ISTJ", "ISFJ", "INFJ", "INTJ", 
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP", 
    "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]

DATASET_MBTI_MAPPING = {
    "biology": ["INTP", "ESTJ", "ENTJ", "ISFJ", "ENFJ", "ENTP"],
    "business": ["ISTJ", "INTJ", "ENTJ", "ISFJ", "ISFP", "ENFJ"],
    "chemistry": ["ISTJ", "ISTP", "ISFP", "ENFP", "ESTP", "ENTP"],
    "computer science": ["INTJ", "ISTJ", "INFP", "INFJ", "ENFP", "ENFJ"],
    "economics": ["ESTJ", "ENTJ", "ISFP", "ESTP", "ENTP", "ENFJ"],
    "engineering": ["ISTJ", "INTJ", "ENFJ", "INFP", "ESFP", "ESTP"],
    "health": ["INTP", "ESTJ", "ESFJ", "ISFJ", "ESFP", "ENFP"],
    "history": ["INTJ", "ISTP", "INTP", "ENFP", "ESFP", "ENFJ"],
    "law": ["ESTJ", "ESTP", "ISFP", "ISTJ", "ESFJ", "ENFJ"],
    "math": ["ISTJ", "INFJ", "ISFP", "INTP", "ESFP", "ESFJ"],
    "other": ["ENTJ", "INTP", "ISFJ", "ISFP", "ESTJ", "ESFJ"],
    "philosophy": ["ESTJ", "ISTJ", "ENTP", "ENFJ", "ENFP", "ESFP"],
    "physics": ["ESTJ", "INTP", "ISFP", "INFP", "ESFP", "ESTP"],
    "psychology": ["ISTJ", "INTJ", "ESTP", "ISFJ", "ENTP", "ENFJ"]
}

def clean_thinking_tags(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature:float, openai_api_key: str, sleep_time: float, 
                 mbti_type: str = None, mbti_data: dict = None, position: str = None) -> None:
        super(DebatePlayer, self).__init__(model_name, name, temperature, openai_api_key, sleep_time)
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.mbti_type = mbti_type
        self.mbti_data = mbti_data
        self.position = position
        
    def get_personality_description(self):
        if self.mbti_type and self.mbti_data and self.mbti_type in self.mbti_data:
            return self.mbti_data[self.mbti_type].get("Debate", "")
        return ""


class MultiAgentDebate:
    def __init__(self,
            model_name: str=None, 
            temperature: float=0.7, 
            num_players: int=3, 
            save_file_dir: str=None,
            openai_api_key: str=None,
            prompts_path: str=None,
            max_round: int=10,                    
            sleep_time: float=0,
            proponent_mbti: str=None,
            opponent_mbti: str=None,
            mbti_data: dict=None
        ) -> None:

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.save_file_dir = save_file_dir
        self.openai_api_key = openai_api_key
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.proponent_mbti = proponent_mbti
        self.opponent_mbti = opponent_mbti
        self.mbti_data = mbti_data

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")

        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'model_name': model_name,
            'temperature': temperature,
            'num_players': num_players,
            'success': False,
            "question": "",
            "options": "",
            "correct_answer": "",
            "Final consensus answer": '',
            "Reason": '',
            'players': {},
            'dialog_history': "",
            'proponent_mbti': proponent_mbti,
            'opponent_mbti': opponent_mbti,
            'proponent_personality': "",
            'opponent_personality': "",
            'rounds_completed': 0,
        }
        prompts = json.load(open(prompts_path))
        self.save_file.update(prompts)
        self.init_prompt()

        self.creat_agents()
        self.init_agents()
    
    def init_prompt(self):
        def prompt_replace(key):
            if key not in self.save_file:
                return
                
            content = self.save_file[key]
            
            if "##question##" in content:
                content = content.replace("##question##", self.save_file.get("question", ""))
            if "##options##" in content:
                content = content.replace("##options##", self.save_file.get("options", ""))
            if "##proponent_personality##" in content:
                proponent_desc = self.mbti_data.get(self.proponent_mbti, {}).get("Debate", "") if self.mbti_data else ""
                content = content.replace("##proponent_personality##", proponent_desc)
            if "##opponent_personality##" in content:
                opponent_desc = self.mbti_data.get(self.opponent_mbti, {}).get("Debate", "") if self.mbti_data else ""
                content = content.replace("##opponent_personality##", opponent_desc)
            if "##proponent_mbti##" in content:
                content = content.replace("##proponent_mbti##", self.proponent_mbti)
            if "##opponent_mbti##" in content:
                content = content.replace("##opponent_mbti##", self.opponent_mbti)
                
            self.save_file[key] = content
        
        prompts_to_process = [
            "proponent_init_prompt", "opponent_init_prompt", "moderator_prompt",
            "proponent_prompt", "opponent_prompt", "proponent_system_prompt", 
            "opponent_system_prompt", "moderator_system_prompt"
        ]
        
        for prompt_key in prompts_to_process:
            if prompt_key in self.save_file:
                prompt_replace(prompt_key)

    def creat_agents(self):
        self.players = [
            DebatePlayer(
                model_name=self.model_name, 
                name=NAME_LIST[0],  
                temperature=self.temperature, 
                openai_api_key=self.openai_api_key, 
                sleep_time=self.sleep_time,
                mbti_type=self.proponent_mbti,
                mbti_data=self.mbti_data,
                position="proponent"
            ),
            DebatePlayer(
                model_name=self.model_name, 
                name=NAME_LIST[1],  
                temperature=self.temperature, 
                openai_api_key=self.openai_api_key, 
                sleep_time=self.sleep_time,
                mbti_type=self.opponent_mbti,
                mbti_data=self.mbti_data,
                position="opponent"
            ),
            DebatePlayer(
                model_name=self.model_name, 
                name=NAME_LIST[2], 
                temperature=self.temperature, 
                openai_api_key=self.openai_api_key, 
                sleep_time=self.sleep_time
            )
        ]
        self.proponent = self.players[0]    
        self.opponent = self.players[1]     
        self.moderator = self.players[2]   

    def init_agents(self):
        self.proponent.set_meta_prompt(self.save_file['proponent_system_prompt'])
        self.opponent.set_meta_prompt(self.save_file['opponent_system_prompt'])
        self.moderator.set_meta_prompt(self.save_file['moderator_system_prompt'])

        print(f"===== Debate Round-1 =====\n")
        print(f"Proponent ({self.proponent_mbti}) vs Opponent ({self.opponent_mbti})\n")
        
        self.proponent.add_event(self.save_file['proponent_init_prompt'])
        self.prop_ans = self.proponent.ask()
        self.prop_ans = clean_thinking_tags(self.prop_ans)
        self.proponent.add_memory(self.prop_ans)

        self.opponent.add_event(self.save_file['opponent_init_prompt'].replace('##proponent response##', self.prop_ans))
        self.opp_ans = self.opponent.ask()
        self.opp_ans = clean_thinking_tags(self.opp_ans)
        self.opponent.add_memory(self.opp_ans)

        self.update_dialog_history("Round-1", self.prop_ans, self.opp_ans)

        self.moderator.add_event(self.save_file['moderator_prompt']\
                                .replace('##dialog_history##', self.save_file['dialog_history']))
        self.mod_ans = self.moderator.ask()
        self.mod_ans = clean_thinking_tags(self.mod_ans)
        self.moderator.add_memory(self.mod_ans)

        self.mod_ans = self.mod_ans.replace('```json', '').replace('```', '').strip()
        try:
            self.mod_ans = json.loads(self.mod_ans)
        except:
            self.mod_ans = {"Is the debate completed": "No", "Final consensus answer": "", "Reason": "Parse error"}

    def update_dialog_history(self, round_name, prop_ans, opp_ans):
        round_history = f"===== {round_name} =====\nProponent ({self.proponent_mbti})：{prop_ans}\nOpponent ({self.opponent_mbti})：{opp_ans}\n"
        self.save_file['dialog_history'] += round_history

    def update_proponent_history(self, round_name, prop_ans):
        round_history = f"===== {round_name} =====\nProponent ({self.proponent_mbti})：{prop_ans}\n"
        self.save_file['dialog_history'] += round_history
        
    def update_opponent_history(self, round_name, opp_ans):
        round_history = f"Opponent ({self.opponent_mbti})：{opp_ans}\n"
        self.save_file['dialog_history'] += round_history

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth',
            5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth',
            9: 'ninth', 10: 'tenth'
        }
        return dct.get(num, f'round-{num}')

    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        
        self.save_file['end_time'] = current_time
        if self.mbti_data:
            self.save_file['proponent_personality'] = self.mbti_data.get(self.proponent_mbti, {}).get("Debate", "")
            self.save_file['opponent_personality'] = self.mbti_data.get(self.opponent_mbti, {}).get("Debate", "")
        
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

    def run(self):
        for round in range(self.max_round - 1):
            if self.mod_ans.get("Is the debate completed", "No") == "Yes":
                break
            else:
                print(f"===== Debate Round-{round+2} =====\n")
                print(f"Proponent ({self.proponent_mbti}) vs Opponent ({self.opponent_mbti})\n")
                
                proponent_prompt = self.save_file['proponent_prompt'] \
                    .replace('##dialog_history##', self.save_file['dialog_history'])\
                    .replace('##round##', self.round_dct(round+2)) \
                    .replace('##resround##', self.round_dct(self.max_round-(round+2)))
                
                self.prop_ans = self.proponent.ask_single_turn(proponent_prompt)
                self.prop_ans = clean_thinking_tags(self.prop_ans)
                self.proponent.add_memory(self.prop_ans)
                self.update_proponent_history(f"Round-{round+2}", self.prop_ans)

                opponent_prompt = self.save_file['opponent_prompt']\
                    .replace('##dialog_history##', self.save_file['dialog_history'])
                self.opp_ans = self.opponent.ask_single_turn(opponent_prompt)
                self.opp_ans = clean_thinking_tags(self.opp_ans)
                self.opponent.add_memory(self.opp_ans)
                self.update_opponent_history(f"Round-{round+2}", self.opp_ans)

                moderator_prompt = self.save_file['moderator_prompt']\
                    .replace('##dialog_history##', self.save_file['dialog_history'])\
                    .replace('##round##', self.round_dct(round+2))

                if round + 2 == self.max_round:
                    moderator_prompt += "\nPlease note: This is the final round. Regardless of whether there are differences, you must make the final judgment!"

                self.mod_ans = self.moderator.ask_single_turn(moderator_prompt)
                self.mod_ans = clean_thinking_tags(self.mod_ans)
                self.moderator.add_memory(self.mod_ans)
                
                self.mod_ans = self.mod_ans.replace('```json', '').replace('```', '').strip()
                try:
                    self.mod_ans = json.loads(self.mod_ans)
                except Exception:
                    self.mod_ans = {"Is the debate completed": "No"}

        if self.mod_ans.get("Is the debate completed", "No") == "Yes":
            self.save_file.update(self.mod_ans)
            correct_answer = self.save_file.get('correct_answer', '')
            final_answer = self.mod_ans.get('Final consensus answer', '')
            self.save_file['success'] = (final_answer == correct_answer)
        else:
            self.save_file["Final consensus answer"] = "No consensus"
            self.save_file["Reason"] = "Debate not completed; no clear consensus reached."
            self.save_file["success"] = False

        for player in self.players:
            self.save_file['players'][player.name] = player.memory_lst

        self.save_file['rounds_completed'] = round + 1

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Input directory containing JSONL files")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output base directory")
    parser.add_argument("-k", "--api-key", type=str, required=True, help="OpenAI api key")
    parser.add_argument("-m", "--model-name", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--mbti-pairs", type=str, help="MBTI pairs to debate (format: ISTJ-ENTP,ISFJ-INTJ)")
    parser.add_argument("--datasets", type=str, help="Specific datasets to process (format: chemistry,computer science,biology)")
    parser.add_argument("--resume", type=int, default=0, help="Resume from debate ID")
    return parser.parse_args()
    
def extract_question_data(data):
    question = data.get("question", "")
    options = data.get("options", [])
    
    options_text = ""
    for i, option in enumerate(options):
        option_label = chr(65 + i) 
        options_text += f"{option_label}. {option}\n"
    
    return {
        "question": question,
        "options": options_text.strip()
    }

def get_dataset_files(input_dir):
    pattern = os.path.join(input_dir, "*.jsonl")
    files = glob.glob(pattern)
    return [f for f in files if os.path.isfile(f)]

def get_dataset_name(file_path):
    filename = os.path.basename(file_path)
    name = filename.replace('.jsonl', '').lower()
    if name.startswith('subset_'):
        name = name[7:] 
    return name

def generate_mbti_pairs_for_dataset(dataset_name):
    dataset_key = dataset_name.lower()
    
    selected_mbtis = None
    for key, mbtis in DATASET_MBTI_MAPPING.items():
        if key in dataset_key:
            selected_mbtis = mbtis
            break
    
    if selected_mbtis is None:
        print(f" Warning: MBTI configuration for dataset '{dataset_name}' not found. Using default configuration.")
        selected_mbtis = ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP"]
    
    print(f" Dataset '{dataset_name}' uses MBTI personalities: {selected_mbtis}")
    
    pairs = []
    
    
    print("Stage 1: Same-MBTI Pairing")
    for mbti in selected_mbtis:
        pairs.append((mbti, mbti))
        print(f"  {mbti} vs {mbti}")
    
    print("Stage 2: Different-MBTI Pairing")
    for i, proponent in enumerate(selected_mbtis):
        for j, opponent in enumerate(selected_mbtis):
            if i != j: 
                pairs.append((proponent, opponent))
                print(f"  {proponent} vs {opponent}")
    
    print(f"Generated {len(pairs)} total pairings.")
    return pairs

def check_existing_results(output_dir, dataset_name, mbti_pairs, num_inputs):
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_output_dir):
        return {}
    
    existing_files = {}
    for debate_id in range(len(mbti_pairs) * num_inputs):
        existing_files[debate_id] = False
        result_file = os.path.join(dataset_output_dir, f"{debate_id}.json")
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('success') is not None: 
                        existing_files[debate_id] = True
            except:
                pass
    return existing_files

def get_task_progress(mbti_pairs, num_inputs, existing_files):
    total_tasks = len(mbti_pairs) * num_inputs
    completed_count = sum(1 for completed in existing_files.values() if completed)
    return total_tasks, completed_count

def process_single_dataset(input_file, output_base_dir, model_name, openai_api_key, mbti_data):
    dataset_name = get_dataset_name(input_file)
    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)

    inputs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    inputs.append(data)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping invalid JSON line: {line[:100]}... Error: {e}")

    mbti_pairs = generate_mbti_pairs_for_dataset(dataset_name)

    existing_files = check_existing_results(output_base_dir, dataset_name, mbti_pairs, len(inputs))
    total_tasks, completed_count = get_task_progress(mbti_pairs, len(inputs), existing_files)
    
    print(" Dataset Statistics:")
    print(f"  Number of questions: {len(inputs)}")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Completed tasks: {completed_count}")
    print(f"  Remaining tasks: {total_tasks - completed_count}")


    if total_tasks == completed_count:
        return completed_count, total_tasks

    pbar = tqdm(total=total_tasks, initial=completed_count, 
                desc=f"{dataset_name[:10]}...", unit="task", ncols=100)

    processed_count = 0
    start_time = time.time()
    
    current_script_path = os.path.abspath(__file__)
    project_path = current_script_path.rsplit("/", 2)[0]
    config_path = f"{project_path}/code/utils/config_qa.json"
    try:
        config = json.load(open(config_path, "r", encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Failed to load configuration file {config_path}: {e}")
        return completed_count, total_tasks

    debate_id = 0
    
    for pair_idx, (proponent_mbti, opponent_mbti) in enumerate(mbti_pairs):
        for data_idx, data in enumerate(inputs):
            if existing_files.get(debate_id, False):
                debate_id += 1
                pbar.update(1)
                continue

            question_data = extract_question_data(data)

            prompts_path = os.path.join(dataset_output_dir, f"{debate_id}-config.json")
            current_config = config.copy()
            current_config.update(question_data)
            current_config['proponent_mbti'] = proponent_mbti
            current_config['opponent_mbti'] = opponent_mbti
            
            current_config['original_question_id'] = data.get('question_id', debate_id)

            with open(prompts_path, "w", encoding="utf-8") as file:
                json.dump(current_config, file, ensure_ascii=False, indent=4)

            debate = None
            try:
                elapsed_time = time.time() - start_time
                if processed_count > 0:
                    avg_time_per_task = elapsed_time / processed_count
                    remaining_tasks = total_tasks - completed_count - processed_count
                    remaining_time = avg_time_per_task * remaining_tasks
                    time_info = f"Estimated remaining time: {remaining_time/3600:.1f}h"
                else:
                    time_info = "Estimating..."
                
                debate_type = "Same-MBTI" if proponent_mbti == opponent_mbti else "Different MBTI"
                pbar.set_description(f"{dataset_name[:8]} {proponent_mbti}{opponent_mbti}{data_idx} [{processed_count+1}/{total_tasks-completed_count}] {time_info}")
                
                debate = MultiAgentDebate(
                    save_file_dir=dataset_output_dir, 
                    num_players=3, 
                    model_name=model_name,
                    openai_api_key=openai_api_key, 
                    prompts_path=prompts_path,
                    temperature=0, 
                    sleep_time=0,
                    max_round=10,
                    proponent_mbti=proponent_mbti,
                    opponent_mbti=opponent_mbti,
                    mbti_data=mbti_data
                )
                debate.run()
                
                if debate:
                    debate.save_file_to_json(debate_id)
                
                processed_count += 1
                
            except Exception as e:
                import traceback
                error_msg = f"[ERROR] Failed to process task {dataset_name}/{debate_id}: {e}"
                print(f"\n{error_msg}")
                traceback.print_exc()
                
                save_data = {
                    'original_question_id': data.get('question_id', debate_id),
                    'question': question_data.get('question', ''),
                    'success': False,
                    'Reason': f"Processing failed: {str(e)}",
                    'start_time': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                    'end_time': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                    'Final consensus answer': '',
                    'rounds_completed': 0,
                    'proponent_mbti': proponent_mbti,
                    'opponent_mbti': opponent_mbti
                }
                save_file_path = os.path.join(dataset_output_dir, f"{debate_id}.json")
                with open(save_file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=4)
                
                processed_count += 1

            debate_id += 1
            pbar.update(1)

    pbar.close()

    dataset_time = time.time() - start_time
    print(f"   Dataset {dataset_name} completed!")
    print(f"   Successfully completed: {processed_count} tasks")
    print(f"   Time elapsed: {dataset_time/60:.1f} minutes")

    
    return completed_count + processed_count, total_tasks

if __name__ == "__main__":
    args = parse_args()
    openai_api_key = args.api_key
    model_name = args.model_name

    current_script_path = os.path.abspath(__file__)
    project_path = current_script_path.rsplit("/", 2)[0]

    mbti_path = f"{project_path}/mbti.json"
    mbti_data = load_mbti_data(mbti_path)

    dataset_files = get_dataset_files(args.input_dir)
    
    if args.datasets:
        target_datasets = [d.strip().lower() for d in args.datasets.split(',')]
        dataset_files = [f for f in dataset_files if get_dataset_name(f) in target_datasets]
    
    if not dataset_files:
        print(f"[ERROR] No JSONL files found in directory {args.input_dir}")
        exit(1)

    print(f"\n Found {len(dataset_files)} datasets:")
    for dataset_file in dataset_files:
        dataset_name = get_dataset_name(dataset_file)
        print(f"  - {dataset_name}")

    total_start_time = time.time()
    total_processed = 0
    total_tasks_all = 0

    for dataset_file in dataset_files:
        dataset_name = get_dataset_name(dataset_file)
        processed, total = process_single_dataset(
            dataset_file, args.output_dir, model_name, openai_api_key, mbti_data
        )
        total_processed += processed
        total_tasks_all += total

    total_time = time.time() - total_start_time
    print(f"\n All datasets processed successfully!")
    print(f"  Overall Statistics:")
    print(f"  Number of datasets: {len(dataset_files)}")
    print(f"  Total tasks: {total_tasks_all}")
    print(f"  Total completed: {total_processed}")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Average time per task: {total_time/total_processed:.1f} seconds")
    print(f"  Results saved to: {args.output_dir}")

    
    print("\n Output Directory Structure:")
    for dataset_file in dataset_files:
        dataset_name = get_dataset_name(dataset_file)
        dataset_dir = os.path.join(args.output_dir, dataset_name)
        if os.path.exists(dataset_dir):
            file_count = len([f for f in os.listdir(dataset_dir) if f.endswith('.json')])
            print(f"  {dataset_name}/: {file_count} files")