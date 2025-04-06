

def visualize_token_to_vis_token_attn_scores(avg_last_token_to_all_image_token_attn_scores, fn_title, output_fn):
    import matplotlib.pyplot as plt

    token_idx = range(len(avg_last_token_to_all_image_token_attn_scores))
    fig, ax = plt.subplots()
    ax.plot(token_idx, avg_last_token_to_all_image_token_attn_scores)

    ax.set(xlabel='Image Token Index', ylabel='Attn Score',
           title=fn_title)
    ax.grid()

    fig.savefig(output_fn)
