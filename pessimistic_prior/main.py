from env import CartPoleStandUp
from agents import DQNSolver, PessDQNSolver

from utils.plotting import plot_scores, plot_queries
from utils import smooth_over, MyParser


def parse_args():

    parser = MyParser()

    parser.add_argument(
        "--outdir", type=str, required=True, default=None,
        help="If supplied, model checkpoints will be saved so "
             "training can be restarted later")

    parser.add_argument(
        "--train", dest="train", type=int, default=0, 
        help="number of episodes to train")

    parser.add_argument(
        "--show", action="store_true", 
        help="Shows a gif of the agent acting under its current set of "
             "parameters")

    parser.add_argument(
        "--example", action="store_true", 
        help="Shows an example gif of the environment with random actions")
    
    parser.add_argument(
        "--render", action="store_true", 
        help="Whether to render the env as we go")

    parser.add_argument(
        "--plot", action="store_true", 
        help="Whether to plot the experiment output")

    parser.add_argument(
        "--model", type=str, default='dqn',
        choices=['dqn', 'pess_dqn'], help="The model to be run."
    )

    return parser.parse_args()


def get_model(model_name, env, outdir, args_dict):

    std_args = (outdir, env)

    arg_agent = {
        'dqn': DQNSolver,
        'pess_dqn': PessDQNSolver,
    }

    model_name = model_name.lower()
    if model_name in arg_agent:
        agent = arg_agent[model_name](*std_args, **args_dict)
    else:
        raise ValueError("Need to specify a model in valid choices.")

    return agent


if __name__ == "__main__":

    args = parse_args()
    cart = CartPoleStandUp(
        score_target=195.,
        episodes_threshold=100,
        reward_on_fail=-10.,
    )
    # cart.get_spaces(registry=False)  # just viewing

    args_dict = {
        "epsilon": None,
    }

    agent = get_model(args.model, cart, args.outdir, args_dict)
    print("RUNNING AGENT", agent)

    if args.example:
        cart.do_random_runs(episodes=1, steps=99, verbose=True)
    
    if args.train:
        solved = agent.solve(
            args.train, verbose=True, render=args.render)
        
        print("\nSolved:", solved, "on step", agent.solved_on, 
              "- time elapsed:", agent.elapsed_time)

    if args.show:
        agent.show()

    if args.plot:
        plot_scores(
            agent.experiment_dir + "scores.png", 
            agent.scores)

        for smooth_over_x in (10, cart.episodes_threshold):
            smoothed_scores = smooth_over(agent.scores, smooth_over_x)
            smooth_title = "smoothed over " + str(smooth_over_x)
            save_loc = (agent.experiment_dir + "smooth_scores_" + 
                str(smooth_over_x) + ".png")
            plot_scores(save_loc, smoothed_scores, title=smooth_title)

        if hasattr(agent, "queries"):
            save_queries = agent.experiment_dir + "queries.png"
            plot_queries(save_queries, agent.queries, agent.total_t)
