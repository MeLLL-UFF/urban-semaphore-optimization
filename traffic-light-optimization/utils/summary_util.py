import os

import numpy as np
import pandas as pd

import matplotlib as mlp
mlp.use("agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, MaxNLocator, FormatStrFormatter)
from matplotlib.lines import Line2D

from definitions import ROOT_DIR
from algorithm.sumo_based.definitions import ROOT_DIR as SUMO_BASED_ROOT_DIR


font = {'size': 24}
mlp.rc('font', **font)


def consolidate_reward(reward_each_step, save_path, name_base):

    reward_df = pd.DataFrame({'reward': reward_each_step})
    reward_df.to_csv(save_path + "/" + name_base + "-" + 'reward' + ".csv")


    f, ax = plt.subplots(1, 1, figsize=(20, 9), dpi=100)
    
    ax.margins(0.05)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

    ax.plot(reward_df, linewidth=2, color='k')
    ax.set_title('reward' + ' - ' + str(np.round(reward_df.mean()[0], decimals=2)))
    plt.savefig(save_path + "/" + name_base + "-" + 'reward' + ".png")
    plt.close()


def consolidate_time_loss(time_loss_each_step, save_path, name_base,
                          baseline_comparison=False, scenario=None, traffic_level_configuration=None,
                          mean=False):

    time_loss_df = pd.DataFrame({'time_loss': time_loss_each_step})
    time_loss_df.to_csv(save_path + "/" + name_base + "-" + 'time_loss' + ".csv")


    f, ax = plt.subplots(1, 1, figsize=(20, 9), dpi=100)
    
    ax.margins(0.05)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

    tail_length = 10
    time_loss_tail = time_loss_df.iloc[-tail_length:]
    final_time_loss = np.round(np.mean(time_loss_tail[time_loss_tail > 0]), decimals=2)[0]

    if mean:
        plot_label = 'frap' + ' ' + '(' + str(final_time_loss) + ')'
    else:
        plot_label = 'frap'

    ax.plot(time_loss_df, linewidth=2, color='k', label=plot_label)

    if baseline_comparison:

        right_on_red_type = 'right_on_red'
        right_on_red_name_base = scenario + '__' + right_on_red_type + '__' + traffic_level_configuration
        right_on_red_records_folder = os.path.join(
            SUMO_BASED_ROOT_DIR, 'records', right_on_red_type, right_on_red_name_base, 'test')

        unregulated_type = 'unregulated'
        unregulated_name_base = scenario + '__' + unregulated_type + '__' + traffic_level_configuration
        unregulated_records_folder = os.path.join(
            SUMO_BASED_ROOT_DIR, 'records', unregulated_type, unregulated_name_base, 'test')

        right_on_red_result_file = right_on_red_records_folder + '/' + right_on_red_name_base + '-test-time_loss.csv'
        unregulated_result_file = unregulated_records_folder + '/' + unregulated_name_base + '-test-time_loss.csv'

        if os.path.isfile(right_on_red_result_file):
            right_on_red_df = pd.read_csv(right_on_red_result_file)
            
            if mean:
                data = right_on_red_df['time_loss'].mean()
                ax.plot([0, time_loss_df.shape[0]], [data, data], linewidth=2, linestyle=':', color='r',
                        label='right on red' + ' ' + '(' + str(np.round(data, decimals=2)) + ')')
            else:
                data = right_on_red_df['time_loss']
                ax.plot(data, linewidth=2, linestyle=':', color='r', label='right on red')

        if os.path.isfile(unregulated_result_file):
            unregulated_df = pd.read_csv(unregulated_result_file)

            if mean:
                data = unregulated_df['time_loss'].mean()
                ax.plot([0, time_loss_df.shape[0]], [data, data], linewidth=2, linestyle=':', color='g',
                        label='unregulated' + ' ' + '(' + str(np.round(data, decimals=2)) + ')')
            else:
                data = unregulated_df['time_loss']
                ax.plot(data, linewidth=2, linestyle=':', color='g', label='unregulated')

        ax.legend()

    ax.set_title('time loss' + ' - ' + str(np.round(time_loss_df.mean()[0], decimals=2)))
    plt.savefig(save_path + "/" + name_base + "-" + 'time_loss' + ".png")
    plt.close()


def consolidate_occupancy_and_speed_inflow_outflow(relative_occupancy_each_step, relative_mean_speed_each_step,
                                                   movements, movement_to_connection, save_path, name_base):

    relative_occupancy_df = pd.DataFrame(relative_occupancy_each_step)
    relative_mean_speed_df = pd.DataFrame(relative_mean_speed_each_step)

    relative_occupancy_df = relative_occupancy_df.rolling(10, min_periods=1).mean()
    relative_mean_speed_df = relative_mean_speed_df.rolling(10, min_periods=1).mean()

    from_lane_set = set()
    to_lane_set = set()

    per_movement_df = pd.DataFrame()
    for movement in movements:

        connection = movement_to_connection[movement]

        from_lane = connection['from'] + '_' + connection['fromLane']
        to_lane = connection['to'] + '_' + connection['toLane']

        per_movement_df.loc[:, movement + '_' + 'inflow' + '_' + 'relative_occupancy'] = \
            relative_occupancy_df.loc[:, from_lane]
        per_movement_df.loc[:, movement + '_' + 'outflow' + '_' + 'relative_occupancy'] = \
            relative_occupancy_df.loc[:, to_lane]
        per_movement_df.loc[:, movement + '_' + 'inflow' + '_' + 'relative_mean_speed'] = \
            relative_mean_speed_df.loc[:, from_lane]
        per_movement_df.loc[:, movement + '_' + 'outflow' + '_' + 'relative_mean_speed'] = \
            relative_mean_speed_df.loc[:, to_lane]

        from_lane_set.add(from_lane)
        to_lane_set.add(to_lane)

    per_movement_df.to_csv(save_path + '/' + name_base + '-' +
                           'occupancy_and_speed_inflow_outflow' + '-' + 'per_movement' + '.csv')


    edge_dict = {}
    lanes = relative_occupancy_df.columns.tolist()
    for lane in lanes:
        edge = lane.split('_')[0]

        if edge in edge_dict:
            edge_dict[edge].append(lane)
        else:
            edge_dict[edge] = [lane]

    per_edge_df = pd.DataFrame()   
    edges = list(edge_dict.keys()) 
    for edge in edges:
        per_edge_df.loc[:, edge + '_' + 'relative_occupancy'] = \
            relative_occupancy_df.loc[:, edge_dict[edge]].mean(axis=1)
        per_edge_df.loc[:, edge + '_' + 'relative_mean_speed'] = \
            relative_mean_speed_df.loc[:, edge_dict[edge]].mean(axis=1)

    per_edge_df.to_csv(save_path + '/' + name_base + '-' +
                       'occupancy_and_speed_inflow_outflow' + '-' + 'per_edge' + '.csv')


    from_lanes_relative_occupancy_df = relative_occupancy_df.loc[:, from_lane_set].mean(axis=1)
    to_lanes_relative_occupancy_df = relative_occupancy_df.loc[:, to_lane_set].mean(axis=1)
    from_lanes_relative_mean_speed_df = relative_mean_speed_df.loc[:, from_lane_set].mean(axis=1)
    to_lanes_relative_mean_speed_df = relative_mean_speed_df.loc[:, to_lane_set].mean(axis=1)

    all_lanes_df = pd.concat(
        [
            from_lanes_relative_occupancy_df, 
            to_lanes_relative_occupancy_df,
            from_lanes_relative_mean_speed_df, 
            to_lanes_relative_mean_speed_df
        ], axis=1, sort=False)

    all_lanes_df.columns = [
        'from_lane_relative_occupancy', 
        'to_lane_relative_occupancy', 
        'from_lane_relative_mean_speed',
        'to_lane_relative_mean_speed'
    ]

    all_lanes_df.to_csv(save_path + '/' + name_base + '-' +
                        'occupancy_and_speed_inflow_outflow' + '-' + 'all_lanes' + '.csv')

    
    f, axs = plt.subplots(len(edges), 2, figsize=(120, 18*len(edges)), dpi=100)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.05, hspace=0.05)

    for i in range(0, len(edges)):

        axs[i][0].margins(0)
        axs[i][1].margins(0)

        edge = edges[i]

        axs[i][0].plot(per_edge_df.iloc[:, i*2+0:i*2+1], linewidth=2, color='k')
        axs[i][1].plot(per_edge_df.iloc[:, i*2+1:i*2+2], linewidth=2, color='k')
        axs[i][0].set_title(edge + ' ' + 'relative_occupancy (%)')
        axs[i][1].set_title(edge + ' ' + 'relative_speed (%)')
        axs[i][0].set_ylim(0, 1)
        axs[i][1].set_ylim(0, 1)

        axs[i][0].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axs[i][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i][0].yaxis.set_minor_locator(MultipleLocator(0.05))

        axs[i][0].xaxis.set_major_locator(MaxNLocator(nbins=12))
        axs[i][0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i][0].xaxis.set_minor_locator(MultipleLocator(10))

        axs[i][1].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axs[i][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i][1].yaxis.set_minor_locator(MultipleLocator(0.05))

        axs[i][1].xaxis.set_major_locator(MaxNLocator(nbins=12))
        axs[i][1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i][1].xaxis.set_minor_locator(MultipleLocator(10))

        axs[i][0].set_axisbelow(True)
        axs[i][0].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

        axs[i][1].set_axisbelow(True)
        axs[i][1].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

    plt.savefig(save_path + '/' + name_base + '-' +
                'occupancy_and_speed_inflow_outflow' + '-' + 'per_edge' + '.png')
    plt.close()


    f, axs = plt.subplots(len(movements), 2, figsize=(120, 18*len(movements)), dpi=100)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.05, hspace=0.05)

    for i in range(0, len(movements)):

        movement = movements[i]

        axs[i][0].margins(0)
        axs[i][1].margins(0)

        axs[i][0].plot(per_movement_df.iloc[:, i*4+0:i*4+1], linewidth=2, color='b')
        axs[i][0].plot(per_movement_df.iloc[:, i*4+1:i*4+2], linewidth=2, color='r')
        axs[i][1].plot(per_movement_df.iloc[:, i*4+2:i*4+3], linewidth=2, color='b')
        axs[i][1].plot(per_movement_df.iloc[:, i*4+3:i*4+4], linewidth=2, color='r')
        axs[i][0].set_title(movement + ' ' + 'relative_occupancy (%)')
        axs[i][1].set_title(movement + ' ' + 'relative_speed (%)')
        axs[i][0].set_ylim(0, 1)
        axs[i][1].set_ylim(0, 1)
        axs[i][0].legend(['inflow', 'outflow'], loc='upper right')
        axs[i][1].legend(['inflow', 'outflow'], loc='upper right')

        axs[i][0].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axs[i][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i][0].yaxis.set_minor_locator(MultipleLocator(0.05))

        axs[i][0].xaxis.set_major_locator(MaxNLocator(nbins=12))
        axs[i][0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i][0].xaxis.set_minor_locator(MultipleLocator(10))

        axs[i][1].yaxis.set_major_locator(MaxNLocator(nbins=5))
        axs[i][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i][1].yaxis.set_minor_locator(MultipleLocator(0.05))

        axs[i][1].xaxis.set_major_locator(MaxNLocator(nbins=12))
        axs[i][1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i][1].xaxis.set_minor_locator(MultipleLocator(10))

        axs[i][0].set_axisbelow(True)
        axs[i][0].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

        axs[i][1].set_axisbelow(True)
        axs[i][1].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

    plt.savefig(save_path + '/' + name_base + '-' +
                'occupancy_and_speed_inflow_outflow' + '-' + 'per_movement' + '.png')
    plt.close()


    f, axs = plt.subplots(1, 2, figsize=(120, 18), dpi=100)
    plt.subplots_adjust(left=0.02, right=0.98, wspace=0.05, hspace=0.05)

    axs[0].margins(0)
    axs[1].margins(0)

    axs[0].plot(all_lanes_df.iloc[:, 0:1], linewidth=2, color='b')
    axs[0].plot(all_lanes_df.iloc[:, 1:2], linewidth=2, color='r')
    axs[1].plot(all_lanes_df.iloc[:, 2:3], linewidth=2, color='b')
    axs[1].plot(all_lanes_df.iloc[:, 3:4], linewidth=2, color='r')
    axs[0].set_title('relative_occupancy (%)')
    axs[1].set_title('relative_speed (%)')
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend(['inflow', 'outflow'], loc='upper right')
    axs[1].legend(['inflow', 'outflow'], loc='upper right')

    axs[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))

    axs[0].xaxis.set_major_locator(MaxNLocator(nbins=12))
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axs[0].xaxis.set_minor_locator(MultipleLocator(10))

    axs[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.05))

    axs[1].xaxis.set_major_locator(MaxNLocator(nbins=12))
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axs[1].xaxis.set_minor_locator(MultipleLocator(10))

    axs[0].set_axisbelow(True)
    axs[0].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

    axs[1].set_axisbelow(True)
    axs[1].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

    plt.savefig(save_path + '/' + name_base + '-' +
                'occupancy_and_speed_inflow_outflow' + '-' + 'all_lanes' + '.png')
    plt.close()


def consolidate_phase_and_demand(absolute_number_of_cars_each_step, traffic_light_each_step,
                                 movements, movement_to_connection, lane_to_traffic_light_index_mapping,
                                 save_path, name_base):

    absolute_number_of_cars_df = pd.DataFrame(absolute_number_of_cars_each_step)
    traffic_light_df = pd.DataFrame(traffic_light_each_step)

    absolute_number_of_cars_per_movement_df = pd.DataFrame()
    for movement in movements:

        connection = movement_to_connection[movement]
        from_lane = connection['from'] + '_' + connection['fromLane']

        absolute_number_of_cars_per_movement_df.loc[:, movement] = absolute_number_of_cars_df.loc[:, from_lane]

    percent_number_of_cars_per_movement_df = absolute_number_of_cars_per_movement_df.div(
        absolute_number_of_cars_per_movement_df.sum(axis=1), 
        axis=0)
    
    absolute_number_of_cars_per_movement_df.to_csv(
        save_path + '/' + name_base + '-' +
        'phase_and_demand' + '-' + 'absolute' + '-' + 'per_movement' + '.csv')

    percent_number_of_cars_per_movement_df.to_csv(
        save_path + '/' + name_base + '-' +
        'phase_and_demand' + '-' + 'percent' + '-' + 'per_movement' + '.csv')
    

    traffic_light_color_mapping = {
        'r': (1, 0, 0),             # red
        'y': (1, 1, 0),             # yellow
        'g': (0, 0.702, 0),         # dark green
        'G': (0, 1, 0),             # green
        's': (0.502, 0, 0.502),     # purple
        'u': (1, 0.502, 0),         # orange
        'o': (0.502, 0.251, 0),     # brown
        'O': (0, 1, 1)              # cyan
    }

    movement_to_traffic_light_index_mapping = {}
    for movement in movements:

        connection = movement_to_connection[movement]
        from_lane = connection['from'] + '_' + connection['fromLane']

        movement_to_traffic_light_index_mapping[movement] = lane_to_traffic_light_index_mapping[from_lane]

    desired_indices = np.array(list(movement_to_traffic_light_index_mapping.values())).astype(np.int)

    # split each traffic light and retrieve desired movements
    traffic_light_df = pd.DataFrame(
        list(map(
            lambda x: np.array(list(x))[desired_indices], 
            traffic_light_df.iloc[:, 0]
        )),
        columns=movement_to_traffic_light_index_mapping.keys()
    )

    traffic_light_df.to_csv(save_path + '/' + name_base + '-' +
                            'phase_and_demand' + '-' + 'traffic_light' + '.csv')

    traffic_light_colors = traffic_light_df.applymap(lambda x: traffic_light_color_mapping[x])


    legend_label_keys_kwargs = {
        'marker': 's', 
        'markersize': 30,
        'linewidth': 0,
    }

    legend_label_keys = [
        Line2D([], [], color=traffic_light_color_mapping['r'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['y'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['g'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['G'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['s'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['u'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['o'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['O'], **legend_label_keys_kwargs)
    ]

    legend_label_values = [
        'red light', 
        'amber (yellow) light', 
        'green light, no priority', 
        'green light, priority', 
        'right on red light', 
        'red+yellow light',
        'off, blinking',
        'off, no signal'
    ]

    x_len, y_len = percent_number_of_cars_per_movement_df.shape

    f, axs = plt.subplots(y_len, 1, figsize=(60, 3*y_len), dpi=100, sharex=True)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.05, hspace=0.1)

    x = range(0, x_len)
    for i in range(0, y_len):

        data = percent_number_of_cars_per_movement_df.iloc[:, i]
        
        axs[i].margins(0)
        axs[i].set_ylim(0, 1)

        if i == y_len - 1:
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i].yaxis.set_minor_locator(MultipleLocator(0.05))

        axs[i].xaxis.set_major_locator(MultipleLocator(120))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.set_minor_locator(MultipleLocator(10))

        axs[i].set_axisbelow(True)
        axs[i].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')


        lane_traffic_light_colors = traffic_light_colors.iloc[:, i]

        axs[i].plot(data, linewidth=2, color='k')
        axs[i].bar(x, data, width=1, color=tuple(lane_traffic_light_colors))

        axs[i].set_ylabel(data.name.split('_', 1)[0], labelpad=-100, rotation=0)

    axs[0].legend(legend_label_keys, legend_label_values, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                  ncol=len(legend_label_values), mode="expand", borderaxespad=0.)

    plt.suptitle('phase and demand distribution (%, in decimal)')

    plt.savefig(save_path + '/' + name_base + '-' +
                'phase_and_demand' + '-' + 'percent' + '-' + 'per_movement' + '.png')
    plt.close()


    x_len, y_len = absolute_number_of_cars_per_movement_df.shape

    f, axs = plt.subplots(y_len, 1, figsize=(60, 3*y_len), dpi=100, sharex=True)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.05, hspace=0.1)

    max_number_of_cars = absolute_number_of_cars_per_movement_df.values.max()

    x = range(0, x_len)
    for i in range(0, y_len):

        data = absolute_number_of_cars_per_movement_df.iloc[:, i]
        
        axs[i].margins(0)
        axs[i].set_ylim(0, max_number_of_cars)

        if i == y_len - 1:
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].yaxis.set_minor_locator(MultipleLocator(1))

        axs[i].xaxis.set_major_locator(MultipleLocator(120))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.set_minor_locator(MultipleLocator(10))

        axs[i].set_axisbelow(True)
        axs[i].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')


        lane_traffic_light_colors = traffic_light_colors.iloc[:, i]

        axs[i].plot(data, linewidth=2, color='k')
        axs[i].bar(x, data, width=1, color=tuple(lane_traffic_light_colors))

        axs[i].set_ylabel(data.name.split('_', 1)[0], labelpad=-100, rotation=0)

    axs[0].legend(legend_label_keys, legend_label_values, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                  ncol=len(legend_label_values), mode="expand", borderaxespad=0.)

    plt.suptitle('phase and demand distribution (absolute numbers)')

    plt.savefig(save_path + '/' + name_base + '-' +
                'phase_and_demand' + '-' + 'absolute' + '-' + 'per_movement' + '.png')
    plt.close()


    f, ax = plt.subplots(1, 1, figsize=(40, 20), dpi=100)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.90, wspace=0.05, hspace=0.05)

    ax.margins(0.05)
    ax.set_ylim(0, 1)

    ax.yaxis.set_major_locator(MaxNLocator(nbins= 10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    x_ticks = []
    width = 1

    sort_order = list(traffic_light_color_mapping.keys())

    accumulated_width = 0
    for i in range(0, traffic_light_df.shape[1]):

        value_counts = traffic_light_df.iloc[:, i].value_counts()
        data = value_counts.div(value_counts.sum())
        sorted_index = sorted(list(data.index), key=lambda x: sort_order.index(x))
        data = data.reindex(sorted_index)
        data = data.cumsum()
        data = data.reindex(index=data.index[::-1])

        lane_traffic_light_colors = [traffic_light_color_mapping[index] for index in data.index]
        
        bar_position = accumulated_width + width/2

        ax.bar(bar_position, data, width=width, color=tuple(lane_traffic_light_colors))

        x_ticks.append(accumulated_width + width/2)
        
        accumulated_width += width + width


    ax.set_xticks(x_ticks)
    ax.set_xticklabels(traffic_light_df.columns)

    ax.set_axisbelow(True)
    ax.grid(axis='y', color='gray', linestyle='dashed', alpha=0.7, which='both')

    ax.legend(legend_label_keys, legend_label_values, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=len(legend_label_values), mode="expand", borderaxespad=0.)

    plt.suptitle('traffic light distribution (%, in decimal)')

    plt.savefig(save_path + '/' + name_base + '-' + 'traffic light distribution' + '-' + 'per_movement' + '.png')
    plt.close()
