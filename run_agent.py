# Imports:
from env import Environment
from agent_brain import QLearningTable
import concurrent.futures
import warnings
import random
import rtamt
import time
import sys


warnings.simplefilter(action='ignore', category=FutureWarning)


DIM = 10
pixels = 16
EPISODE = 50000
ROB_NORMALIZER = 1
MAX_STEP_PER_EPISODE = 100


spec1 = rtamt.StlDiscreteTimeSpecification()
spec1.declare_var('x', 'float')
spec1.declare_var('y', 'float')


spec1.unit = 'ms'
spec1.set_sampling_period(100, 'ms', 0.1)

spec1.spec = ' (eventually[0,0.5] ( (y==2 and x==4) and eventually[0.0,0.5]  (x>=9 and y>=9) )  )'


try:
    spec1.parse()
    spec1.pastify()

except rtamt.RTAMTException as err:
    print('RTAMT Exception: {}'.format(err))
    sys.exit()


def update():
    start_time = time.time()

    global flag

    # maze1 = [ [0]*DIM for i in range(DIM) ]

    # Resulted list for the plotting Episodes via Steps
    steps = []

    # Summed costs for all episodes in resulted list
    all_costs = []

    for episode in range(EPISODE):
        # Initial Observation
        observation = env.reset()

        # Updating number of Steps for each Episode
        i = 0

        # Updating the cost for each episode
        cost = 0

        prev_rob = None

        # maze2 = [ [0]*DIM for i in range(DIM) ]
        for _ in range(MAX_STEP_PER_EPISODE):
            
            # Refreshing environment
            env.render()

            # RL chooses action based on observation
            action = RL.choose_action(str(observation))

            addon = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            ad = random.choice(addon)
            probabilities = [0.8 + ad, (1 - (0.8 + ad))/2, (1 - (0.8 + ad))/2]

            if action == 0:   #vertical
                numbers = [0, 2, 3]
                action = random.choices(numbers, probabilities)[0]

            elif action == 1:   #vertical
                numbers = [1, 2, 3]
                action = random.choices(numbers, probabilities)[0]

            elif action == 2:   #horizontal
                numbers = [2, 0, 1]
                action = random.choices(numbers, probabilities)[0]

            else:   #horizontal
                numbers = [3, 0, 1]
                action = random.choices(numbers, probabilities)[0]

            next_agent_pos, collision = env.check_pos(action)
            if collision:
                observation_, reward, done, next_pos, stop = 'dyn_pos', 0, True, next_agent_pos, False
            else:
                # RL takes an action and get the next observation and reward
                observation_, reward, done, next_pos, stop = env.step(action)

            # # stl
            #variables like x,y will come from next_step

            if type(observation_) == type('str') and observation_ != 'dyn_obs':
                pass
            else:
                x = next_pos[0]/pixels
                y = next_pos[1]/pixels


                rob1 = spec1.update(0, [('x', x), ('y', y)])

                if prev_rob is not None:
                    rob1 = rob1 - prev_rob

                reward += (rob1/ROB_NORMALIZER)

                prev_rob = rob1


            # RL learns from this transition and calculating the cost
            cost += RL.learn(str(observation), action, reward, str(observation_))

            # Swapping the observations - current and next
            observation = observation_

            # Calculating number of Steps in the current Episode
            i += 1


            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                steps += [i]
                all_costs += [cost]
                break

        print(f'\nEpisode: {episode}:\n')


        if stop:
            end_time = time.time()
            time_taken = end_time - start_time
            final_stat(steps, all_costs, RL, env)
            print(f'Time taken: {time_taken:.2f} s')
            return

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Time taken: {time_taken:.2f} s')
    final_stat(steps, all_costs, RL, env)


def final_stat(steps, all_costs, RL, env):
    # Showing the final route
    env.final()

    # Showing the Q-table with values for each action
    RL.print_q_table()

    # Plotting the results
    RL.plot_results(steps, all_costs)




# Calling for the environment
env = Environment()
    
# Calling for the main algorithm
RL = QLearningTable(actions=list(range(env.n_actions)))
# Running the main loop with Episodes by calling the function update()
env.after(100, update)  # Or just update()
# update(RL, env)

with concurrent.futures.ThreadPoolExecutor() as executor:
    f1 = executor.submit(env.cat1_move)
    env.mainloop()