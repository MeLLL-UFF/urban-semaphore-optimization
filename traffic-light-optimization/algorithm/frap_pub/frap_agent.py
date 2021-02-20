
import numpy as np

from keras.layers import Input, Dense, Flatten, Reshape, Layer, Lambda, RepeatVector, Activation, Embedding, Conv2D
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers.merge import concatenate, add, dot, maximum, multiply
from keras import backend as K

from algorithm.frap_pub.network_agent import NetworkAgent, conv2d_bn, Selector


def slice_tensor(x, index):
    x_shape = K.int_shape(x)
    if len(x_shape) == 3:
        return x[:, index, :]
    elif len(x_shape) == 2:
        return Reshape((1, ))(x[:, index])



def cal_lane_demand(num_vec, phase, feature_shape, dic_agent_conf):
    # process inputs
    expand_cur_phase = Reshape((1, feature_shape["current_phase"]), name="reshaped_phase")(phase)
    reshaped_num_vec = Reshape((1, feature_shape["lane_num_vehicle"]), name="reshaped_vec_num")(num_vec)
    concat_feature = concatenate([reshaped_num_vec, expand_cur_phase], axis=1, name="concat_feature")

    # shared dense layers
    shared_dense1 = Dense(dic_agent_conf["D_DENSE"], activation="relu", name="shared_dense1")
    shared_dense2 = Dense(dic_agent_conf["D_DENSE"], activation="relu", name="shared_dense2")
    shared_dense3 = Dense(1, activation="linear", name="shared_dense3")

    # lane demand
    list_lane_demand = []
    for i in range(feature_shape["lane_num_vehicle"]):
        locals()["lane_%d" % i] = Lambda(slice_tensor, arguments={'index': i}, name="lane_%d" % i)(concat_feature)
        locals()["p_%d" % i] = shared_dense3(shared_dense2(shared_dense1(locals()["lane_%d" % i])))
        list_lane_demand.append(locals()["p_%d" % i])

    lane_demand = concatenate(list_lane_demand, name="lane_demand")

    return lane_demand


def conflict_matrix(x, num_actions):
    # define conlict matrix
    if num_actions == 4:
        c = K.constant([[0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 1, 0, 0]])
    elif num_actions == 2:
        c = K.constant([[1, 0], [1, 0], [0, 1], [0, 1]])
    return K.dot(x, c)


def _share_cooperate_network(x):
    return x


def _share_compete_network(x):
    return x

def compete_max(x):
    return K.max(x, axis=1)

def share_compete_network(p, dic_agent_conf, ind):
    # share_competing network
    print("&& ", len(K.int_shape(p)), K.int_shape(p))
    dense1 = Dense(dic_agent_conf["D_DENSE"], activation="relu", name="dense1_%d"%ind)(p)
    dense2 = Dense(dic_agent_conf["D_DENSE"], activation="relu", name="dense2_%d"%ind)(dense1)
    q_values = Dense(1, activation="linear", name="q_values_%d"%ind)(dense2)
    return q_values

def competing_network(p, dic_agent_conf, num_actions):
    # competing network
    dense1 = Dense(dic_agent_conf["D_DENSE"], activation="relu", name="dense1")(p)
    dense2 = Dense(dic_agent_conf["D_DENSE"], activation="relu", name="dense2")(dense1)
    q_values = Dense(num_actions, activation="linear", name="q_values")(dense2)
    return q_values

def relation(x, dic_traffic_env_conf):
    relations = []
    phases = dic_traffic_env_conf["UNIQUE_PHASE"]
    for phase_1 in phases:
        zeros = [0]*(len(phases) - 1)
        count = 0
        for phase_2 in phases:
            if phase_1 == phase_2:
                continue
            movements_1 = phase_1.split("_")
            movements_2 = phase_2.split("_")
            all_movements = movements_1 + movements_2
            unique_movements = list(set(all_movements))
            zeros[count] = len(all_movements) - len(unique_movements)
            count += 1
        relations.append(zeros)
    relations = np.array(relations).reshape(1, len(phases), len(phases) - 1)
    batch_size = K.shape(x)[0]
    constant = K.constant(relations)
    constant = K.tile(constant, (batch_size, 1, 1))
    return constant


class FrapAgent(NetworkAgent):

    def build_network(self):

        unique_phases = self.dic_traffic_env_conf["UNIQUE_PHASE"]
        max_number_of_movements_per_phase = max(map(lambda x: len(x.split('_')), unique_phases))
        number_of_phases = len(unique_phases)
        unique_movements = self.dic_traffic_env_conf["UNIQUE_MOVEMENT"]
        number_of_movements = len(unique_movements)

        dic_input_node = {}
        feature_shape = {}
        for feature_name in self.dic_traffic_env_conf["STATE_FEATURE_LIST"]:

            vec_feature = False
            if "phase" in feature_name and self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]
                          ["D_" + feature_name.upper()][0] * number_of_movements,)
            elif "phase" in feature_name and not self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                _shape = self.dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()]
            else:
                _shape = (self.dic_traffic_env_conf["DIC_FEATURE_DIM"]
                          ["D_" + feature_name.upper()][0] * number_of_movements,)
                vec_feature = True

            dic_input_node[feature_name] = Input(shape=_shape, name="input_" + feature_name)
            if vec_feature:
                vec_input_node = dic_input_node[feature_name]
            feature_shape[feature_name] = _shape[0]

        p = Activation('sigmoid')(Embedding(2, 4, input_length=number_of_movements)(dic_input_node["current_phase"]))
        d = Dense(4, activation="sigmoid", name="num_vec_mapping")
        dic_lane = {}
        for i, m in enumerate(unique_movements):
            tmp_vec = d(
                Lambda(slice_tensor, arguments={"index": i}, name="vec_%d" % i)(vec_input_node))
            tmp_phase = Lambda(slice_tensor, arguments={"index": i}, name="phase_%d" % i)(p)
            dic_lane[m] = concatenate([tmp_vec, tmp_phase], name="lane_%d" % i)
        list_phase_pressure = []
        lane_embedding = Dense(self.num_actions*2, activation="relu", name="lane_embedding")
        for phase in unique_phases:
            movements = phase.split("_")

            if len(movements) > 1:
                phase_tensor = add([lane_embedding(dic_lane[m]) for m in movements], name=phase)
            else:
                phase_tensor = lane_embedding(dic_lane[movements[0]])

            list_phase_pressure.append(phase_tensor)

        constant = Lambda(relation, arguments={"dic_traffic_env_conf": self.dic_traffic_env_conf},
                        name="constant")(vec_input_node)
        relation_embedding = Embedding(max_number_of_movements_per_phase, 4, name="relation_embedding")(constant)

        # rotate the phase pressure
        if self.dic_agent_conf["ROTATION"]:
            list_phase_pressure_recomb = []
            num_phase = self.num_actions

            for i in range(num_phase):
                for j in range(num_phase):
                    if i != j:
                        list_phase_pressure_recomb.append(
                            concatenate([list_phase_pressure[i], list_phase_pressure[j]],
                                        name="concat_compete_phase_%d_%d" % (i, j)))

            list_phase_pressure_recomb = concatenate(list_phase_pressure_recomb, name="concat_all")
            feature_map = Reshape((number_of_phases, number_of_phases - 1, self.num_actions*2*2))(list_phase_pressure_recomb)
            lane_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu", name="lane_conv")(feature_map)
            if self.dic_agent_conf["MERGE"] == "multiply":
                relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                                       name="relation_conv")(relation_embedding)
                combine_feature = multiply([lane_conv, relation_conv], name="combine_feature")
            elif self.dic_agent_conf["MERGE"] == "concat":
                relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                                       name="relation_conv")(relation_embedding)
                combine_feature = concatenate([lane_conv, relation_conv], name="combine_feature")
            elif self.dic_agent_conf["MERGE"] == "weight":
                relation_conv = Conv2D(1, kernel_size=(1, 1), activation="relu", name="relation_conv")(relation_embedding)
                relation_conv = Lambda(lambda x: K.repeat_elements(x, self.dic_agent_conf["D_DENSE"], 3),
                                       name="expansion")(relation_conv)
                combine_feature = multiply([lane_conv, relation_conv], name="combine_feature")
            hidden_layer = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                                  name="combine_conv")(combine_feature)
            before_merge = Conv2D(1, kernel_size=(1, 1), activation="linear", name="befor_merge")(hidden_layer)
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(Reshape((number_of_phases, number_of_phases - 1))(before_merge))
            # conv2 = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu", name="conv2")(conv1)
            # conv3 = Conv2D(1, kernel_size=(1, 1), activation="linear", name="conv3")(conv2)
            # conv3 = Reshape((8, 7))(conv3)
            # q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(conv3)

        else:
            if self.dic_agent_conf['CONFLICT_MATRIX']:
                phase_pressure = Reshape((vec_input_node,), name="phase_pressure")(phase_pressure)
            else:
                phase_pressure = Reshape((vec_input_node * 2,), name="phase_pressure")(lane_demand)
            q_values = competing_network(phase_pressure, self.dic_agent_conf, self.num_actions)

        network = Model(inputs=[dic_input_node[feature_name]
                                for feature_name in self.dic_traffic_env_conf["STATE_FEATURE_LIST"]],
                        outputs=q_values)
        # network.compile(optimizer=RMSprop(lr=self.dic_agent_conf["LEARNING_RATE"]),
        #                 loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()

        return network
