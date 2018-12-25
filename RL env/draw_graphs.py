import os

def draw_q_values_graphs(log_path,log_name, game_name, num_actions):

    q_lower_values = {}
    q_upper_values = {}
    for action in range(num_actions):
        q_lower_values[action] = []
        q_upper_values[action] = []

    started_relevant_log = False

    def extract_pattern(pattern_='q_values_lower:'):
        import re

        with open(os.path.join(log_path, log_name + '.log'), 'r') as fo:
            text = fo.read()
            text = text.replace('\n', '')
        #print(len(text))

        pattern1 = r"\bq_values_lower\: \[(.*?)\]"
        pattern2 = r"\bq_values_upper\: \[(.*?)\]"

        compiled1 = re.compile(pattern1)
        res_lower = re.findall(compiled1, text)

        res_lower = [re.split(r" *", r) for r in res_lower]
        res_lower = [r[:-1] if r[-1] == '' else r for r in res_lower]
        res_lower = [[float(x) for x in r] for r in res_lower if len(r) == num_actions]



        compiled2 = re.compile(pattern2)
        res_upper = re.findall(compiled2, text)
        res_upper = [re.split(r" *", r) for r in res_upper]
        res_upper = [r[:-1] if r[-1] == '' else r for r in res_upper]
        res_upper = [[float(x) for x in r] for r in res_upper if len(r) == num_actions]



        print(res_lower)
        print(res_upper)
        print("")
        print(len(res_lower))
    extract_pattern()
    '''
    with open(os.path.join(log_path, log_name + '.log'), 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            #print(i)
            if not started_relevant_log:
                if game_name in line:
                    started_relevant_log = True
            else:
                if "q_values_lower" in line or "q_values_upper" in line:
                    info = []
                    info = info.append(line)
                    continue
                    while "INFO" not in line:
                        info.append(line)
                        continue
                print(info)
                break
    '''



def main():
    log_path = "RL env"
    log_name = "rl_AE"
    game_name = "Qbert-v0"
    num_actions = 6
    draw_q_values_graphs(log_path,log_name, game_name, num_actions)


if __name__ == "__main__":
    main()
