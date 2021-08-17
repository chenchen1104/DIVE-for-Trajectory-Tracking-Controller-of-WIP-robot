import time
from solver import FeedForwardModel
import logging
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = False

'''全局参数定义'''
linewidth = 0.5  # 绘图中曲线宽度
fontsize = 5  # 绘图字体大小
markersize = 2.5  # 标志中字体大小
legend_font_size = 5  # 图例中字体大小


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' Created successfully')
        return True
    else:
        print(path + ' Directory already exists')
        return False


def save_figure(dir, name):
    mkdir(dir)
    plt.savefig(dir + name, bbox_inches='tight')


def plot_result(x_list, dot_x_list, x_desire_list, dot_x_desire_list, theta_list, dot_theta_list, theta_desire_list,
                delta_z_list,
                figure_number, n):
    '''绘图参数定义'''
    label = ["integrate forward","integrate backward(ours)"]
    # ["back_constrained_LSTM.pth", "back_unconstrained_LSTM.pth", "back_constrained_FCIMPRO.pth",
    #         "back_unconstrained_FCIMPRO.pth"]
    color = ["r", "g", "b", "y", "coral"]
    line_style = ["-", "-", "-", "-", "-", "-"]
    '''绘制x曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("x $(m)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(x_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)
    plt.plot(x_desire_list[0], label="$x_{target}$", linestyle="--", linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp" + str(n) + "/", "x_Curve.png")
    plt.show()

    '''绘制theta曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("theta $(Degree)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(theta_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)
    plt.plot(theta_desire_list[0], label="$\\theta_{target}$", linestyle="--", linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp" + str(n) + "/", "theta_Curve.png")
    plt.show()

    '''绘制dot_x曲线'''
    plt.figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("dot_x $(m/s)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(dot_x_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)
    plt.plot(dot_x_desire_list[0], label="$x_{target}$", linestyle="--", linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp" + str(n) + "/", "dot_x_Curve.png")
    plt.show()

    '''绘制dot_theta曲线'''
    plt.figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("dot_theta $(Degree/s)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(dot_theta_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp" + str(n) + "/", "dot_theta_Curve.png")
    plt.show()

    '''绘制delta_z曲线'''
    plt.figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("control $(N*m)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(delta_z_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)

    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp" + str(n) + "/", "control_Curve.png")
    plt.show()


def train(config, fbsde):
    logging.basicConfig(level=logging.INFO, format='%(levelname)-6s %(message)s')

    if fbsde.y_init:
        logging.info('Y0_true: %.4e' % fbsde.y_init)

    net = FeedForwardModel(config, fbsde)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr_value, weight_decay=config.weight_decay)
    start_time = time.time()
    best_terminal_loss = float('+inf')
    best_loss = float('+inf')
    totaly = []
    dw_valid = fbsde.sample(config.valid_size)
    for step in range(config.num_iterations + 1):
        if step % config.logging_frequency == 0:
            net.train(False)
            elapsed_time = time.time() - start_time
            if config.back == False:
                loss, y0, ye, totalx, totalu = net(dw_valid)
                init = y0.detach().numpy()
                x_sample = totalx[-1]
                terminal_cost = (torch.mean(ye[:, 0])).detach().numpy()
                totaly.append(init.item())
                logging.info(
                    "step: %5u, loss: %.4e, Y0: %.4e, x: %.4e, dot_x: %.4e, theta: %.4e, dot_theta: %.4e, cost: %.4e, elapsed time: %3u" % (
                        step, loss, init.item(), torch.mean(abs(x_sample[:, 0, 0])), torch.mean(abs(x_sample[:, 1, 0])),
                        torch.mean(abs(x_sample[:, 2, 0])), torch.mean(abs(x_sample[:, 3, 0])),
                        torch.mean(ye[:, 0]), elapsed_time))
            else:
                loss, y0, ye, totalx, totalu = net(dw_valid)
                init = (torch.mean(y0[:, 0])).detach().numpy()
                x_sample = totalx[-1]
                terminal_cost = (torch.mean(ye[:, 0])).detach().numpy()
                totaly.append(init.item())
                logging.info(
                    "step: %5u, loss: %.4e, Y0: %.4e, x: %.4e, dot_x: %.4e, theta: %.4e, dot_theta: %.4e, cost: %.4e, elapsed time: %3u" % (
                        step, loss, init.item(), torch.mean(abs(x_sample[:, 0, 0])), torch.mean(abs(x_sample[:, 1, 0])),
                        torch.mean(abs(x_sample[:, 2, 0])), torch.mean(abs(x_sample[:, 3, 0])),
                        torch.mean(ye[:, 0]), elapsed_time))
            # 根据终止状态代价最小，选择最好的模型
            if loss < best_loss:
                best_loss = loss
                print("model saved to", config.model_save_path)
                torch.save(net, config.model_save_path)
        dw_train = fbsde.sample(config.batch_size)
        net.train(True)
        optimizer.zero_grad()
        if config.back == False:
            loss, init, ye, totalx, totalu = net(dw_train)
        else:
            loss, y0, ye, totalx, totalu = net(dw_valid)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
        optimizer.step()
    return totaly


def valid(config, fbsde, path):
    net = torch.load(path)
    net.eval()
    # dw_valid = fbsde.sample(config.valid_size)
    dw_valid = torch.zeros([config.valid_size, config.dim, fbsde.num_time_interval])
    loss, init, ye, totalx, totalu = net(dw_valid)
    return totalx, totalu


def plot(cfg, fbsde, n):
    path = ["forward_constrained_LSTM.pth","back_constrained_LSTM.pth"]
    # path = ["forward_constrained_LSTM.pth", "forward_unconstrained_LSTM.pth", "forward_constrained_FCIMPRO.pth",
    #         "forward_unconstrained_FCIMPRO.pth", "back_constrained_LSTM.pth", "back_unconstrained_LSTM.pth",
    #         "back_constrained_FCIMPRO.pth", "back_unconstrained_FCIMPRO.pth"]
    x = []
    dot_x = []
    theta = []
    dot_theta = []
    x_desire = []
    dot_x_desire = []
    theta_desire = []
    delta_z = []

    for p in path:
        if "back" in p:
            cfg.back = False
        if "forward" in p:
            cfg.back = True
        if "constrained" in p:
            cfg.constrained = True
        if "unconstrained" in p:
            cfg.constrained = False
        if "LSTM" in p:
            cfg.lstm = True
            cfg.fcsame = False
        if "FCIMPRO" in p:
            cfg.lstm = False
            cfg.fcsame = True

        x_sample, u = valid(cfg, fbsde, p)

        state = []
        for i in range(len(x_sample)):
            state.append(torch.mean(x_sample[i], dim=0))

        x_list = []
        for i in range(len(x_sample)):
            x_list.append(state[i][0])
        x.append(x_list)

        dot_x_list = []
        for i in range(len(x_sample)):
            s = state[i][1].detach().numpy()
            dot_x_list.append(s)
        dot_x.append(dot_x_list)

        theta_list = []
        for i in range(len(x_sample)):
            s = state[i][2].detach().numpy()
            theta_list.append(s)
        theta.append(theta_list)

        dot_theta_list = []
        for i in range(len(x_sample)):
            s = state[i][3].detach().numpy()
            dot_theta_list.append(s)
        dot_theta.append(dot_theta_list)
        delta_z_list = []
        for i in range(len(x_sample)):
            s = u[i][0][0].detach().numpy()
            delta_z_list.append(s)
        delta_z.append(delta_z_list)

        theta_desire_list = []
        for i in range(len(x_sample)):
            theta_desire_list.append(0)
        theta_desire.append(theta_desire_list)

        x_desire_list = []
        for i in range(len(x_sample)):
            t = i * 0.02
            x_desire_list.append((20 * t - t ** 2) * np.exp(-0.5 * t))
        x_desire.append(x_desire_list)

        dot_x_desire_list = []
        for i in range(len(x_sample)):
            t = i * 0.02
            dot_x_desire_list.append((20 - 2 * t) * np.exp(-0.5 * t) - 0.5 * (20 * t - t ** 2) * np.exp(-0.5 * t))
        dot_x_desire.append(dot_x_desire_list)

    plot_result(x, dot_x, x_desire, dot_x_desire, theta, dot_theta, theta_desire, delta_z, len(x), n)


if __name__ == '__main__':
    from config import get_config
    from equation import get_equation

    cfg = get_config('WIP_LINEAR')
    fbsde = get_equation('WIP_LINEAR', cfg.dim, cfg.total_time, cfg.delta_t)
    # True:y0的方差当做损失函数
    # cfg.back = True
    # cfg.constrained = True
    # cfg.lstm = True
    # cfg.fcsame = False
    # total_y1 = train(cfg, fbsde)
    for i in range(1):
        plot(cfg, fbsde, i + 1)
