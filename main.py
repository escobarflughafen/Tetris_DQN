from os import error

from torch.serialization import load
from agent import Agent
from agent import load_and_play 
import argparse
import time
from deep_q_network import DeepQNetwork1 as DeepQNetwork

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=(1e-3))
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.0005)

    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--target_net_update_period", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--memory_size", type=int, default=30000)
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--model_path", type=str, default="models")
    parser.add_argument("--mode", type=str, default='play', help="'play' or 'train'")
    parser.add_argument("--network", type=str, default='./dqnetwork.pth', help="load network for playing")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    if args.mode=="play":
        print('testing mode: press ctrl+c to quit')
        load_and_play(args.network, args)
    
    elif args.mode=="train":
        print('training mode: press ctrl+c to quit')
        agent = Agent(args)
    
        start_epoch = time.time()
        for i in range(agent.batch_size * 20):
            if(i % agent.batch_size == 0):
                print('exploring step:', i)
            agent.epsilon_greedy_play(render=False)
    
        while (True):
            #do_render = agent.epoch >= 100 and (agent.epoch) % (4 * 50) == 0
            while not agent.epsilon_greedy_play(render=False)[-1]:
                pass

            
            agent.train()
            print('time elapsed:',
                  time.time() - start_epoch,
                  '\tresults:', agent.results['scores'][-1], agent.results['dropped_tetrominoes'][-1], agent.results['cleared_lines'][-1])
            print('epsilon:', agent.epsilon)
    
            if ((agent.epoch) % args.save_interval == 0):
                agent.plot_results()
                #agent.save_model()
            
            
    
    else:
        print('error: invaild mode name')

    