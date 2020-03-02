
import city.traffic_light_system.traffic_light.configurer.off.off_traffic_light_configurer \
    as off_traffic_light_configurer
import city.traffic_light_system.traffic_light.configurer.static.static_traffic_light_configurer \
    as static_traffic_light_configurer
import city.traffic_light_system.traffic_light.configurer.actuated.time_gap_actuated_traffic_light_configurer \
    as time_gap_actuated_traffic_light_configurer
import city.traffic_light_system.traffic_light.configurer.actuated.time_loss_actuated_traffic_light_configurer \
    as time_loss_actuated_traffic_light_configurer
import city.traffic_light_system.traffic_light.configurer.sotl.sotl_marching_traffic_light_configurer \
    as sotl_marching_traffic_light_configurer
import city.traffic_light_system.traffic_light.configurer.sotl.sotl_phase_traffic_light_configurer \
    as sotl_phase_traffic_light_configurer
import city.traffic_light_system.traffic_light.configurer.sotl.sotl_platoon_traffic_light_configurer \
    as sotl_phatoon_traffic_light_configurer
import city.traffic_light_system.traffic_light.configurer.sotl.sotl_request_traffic_light_configurer \
    as sotl_request_traffic_light_configurer
import city.traffic_light_system.traffic_light.configurer.sotl.sotl_wave_traffic_light_configurer \
    as sotl_wave_traffic_light_configurer

from experiment.strategy import OFF, STATIC, TIME_GAP_ACTUATED, TIME_LOSS_ACTUATED, SOTL_MARCHING, \
    SOTL_PHASE,  SOTL_PLATOON, SOTL_REQUEST, SOTL_WAVE


traffic_light_configurer_instances = {
    OFF: off_traffic_light_configurer.instance,
    STATIC: static_traffic_light_configurer.instance,
    TIME_GAP_ACTUATED: time_gap_actuated_traffic_light_configurer.instance,
    TIME_LOSS_ACTUATED: time_loss_actuated_traffic_light_configurer.instance,
    SOTL_MARCHING: sotl_marching_traffic_light_configurer.instance,
    SOTL_PHASE: sotl_phase_traffic_light_configurer.instance,
    SOTL_PLATOON: sotl_phatoon_traffic_light_configurer.instance,
    SOTL_REQUEST: sotl_request_traffic_light_configurer.instance,
    SOTL_WAVE: sotl_wave_traffic_light_configurer.instance
}