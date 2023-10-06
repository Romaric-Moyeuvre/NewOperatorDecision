## IMPORT ##########

from shiny import App, reactive, render, ui
from shiny.types import ImgData

import os
import yaml
import sqlite3
import subprocess

####################

div_margin = {"style":"margin:4px;"}
div_space_around = {"style":"display:flex;justify-content:space-around;align-items:center;margin:4px;"}

div_title_style = {"style":"display:flex;justify-content:center;align-items:center;margin:4px;"}

div_dataset_main_container = {"style": "border:solid 1px;border-radius:5px;padding:4px;margin:4px;"}
div_dataset_sub_main_container = {"style": "display:flex;justify-content:space-between;align-items:center;"}
div_dataset_title_status = {"style":"display:flex;justify-content:end;align-items:end;"}
div_dataset_container_button = {"style":"display:flex;justify-content:center;align-items:center;height:50px;"}
div_dataset_button = {"style":"width:115px;margin:auto 5px;"}

div_container_parameters_style = {"style":"overflow-y:scroll;padding:5px;height:340px;"}
div_parameters_style = {"style":"margin-bottom:5px;margin-right:10px;border:solid 1px;border-radius:5px;padding:4px;"}

div_main_container_style = {"style": "display:flex;justify-content:center;align-items:center;border:solid 1px;height:680px;margin:20px 10px 0 10px;"}
div_left_container_style = {"style": "border-right:solid 1px;height:100%;width:200px;"}
div_left_top_container_style = {"style": "display:flex;justify-content:center;align-items:center;height:50px;width:150px;margin:40px 25px;"}
div_left_bottom_container_style = {"style": "display:flex;justify-content:center;align-items:center;height:calc(100% - 130px);"}
div_right_container_style = {"style": "display:flex;justify-content:center;align-items:center;height:100%;width:calc(100% - 200px);"}

div_container_global_parameters_style = {"style":"display:flex;justify-content:center;align-items:center;height:340px;"}

div_display_style = {"style":"height:100%;margin:0 auto;overflow-y:scroll"}

app_ui = ui.page_fluid(
    ui.h2("Operator Decision Modelisation"),
    ui.layout_sidebar(
        ui.panel_sidebar({"style": "height: 850px;"},
            ui.div(
                ui.div(div_title_style,ui.h5("DATASETS")),
                ui.div(div_dataset_main_container,
                    ui.div(div_dataset_sub_main_container,
                        ui.div(div_dataset_title_status,ui.h6({"style": "margin: auto;"},"Measures Dataset")), 
                        ui.div(div_dataset_title_status,ui.div(ui.p({"style": "margin: auto 5px;"}, "Status: ")),ui.div(ui.output_text("measures_dataset_status"))),
                    ),
                    ui.div(div_dataset_container_button,
                        ui.div(div_dataset_button,ui.input_action_button("measures_dataset_create", "CREATE", width="115px")),
                        ui.div(div_dataset_button,ui.input_action_button("measures_dataset_delete", "DELETE", width="115px")),
                    ),
                ),
                ui.div(div_dataset_main_container,
                    ui.div(div_dataset_sub_main_container,
                        ui.div(div_dataset_title_status,ui.h6({"style": "margin: auto;"}, "Stats Dataset")), 
                        ui.div(div_dataset_title_status,ui.div(ui.p({"style": "margin: auto 5px;"}, "Status: ")),ui.div(ui.output_text("stats_dataset_status"))),
                    ),
                    ui.div(div_dataset_container_button,
                        ui.div(div_dataset_button,ui.input_action_button("stats_dataset_create", "CREATE", width="115px")),
                        ui.div(div_dataset_button,ui.input_action_button("stats_dataset_delete", "DELETE", width="115px")),
                    ),
                ),
            ),
            ui.div(div_margin,
                ui.div(div_title_style,ui.h5("GLOBAL PARAMETERS")),
                ui.div(div_title_style,
                    ui.p({"style": "width: 60px;"},"Alarm : "),
                    ui.div(div_margin, ui.input_radio_buttons("alarm_type", "", ["TEMP", "PSAL", "BOTH"], inline=True, width="100%")),
                ),
                ui.div(div_space_around,ui.input_slider("tmult","True Alarm x",1,10,1,step=1,ticks=True,width="115px"),ui.input_slider("fmult","False Alarm x",1,10,1,step=1,ticks=True,width="115px"))
            ),
            ui.navset_tab(
                ui.nav("DTC",
                    ui.div(div_container_parameters_style,
                        ui.div(div_parameters_style,ui.input_select("dtc_criterion","CRITERION",["gini","entropy"])),
                        ui.div(div_parameters_style,ui.input_slider("dtc_max_depth","MAX DEPTH", 1, 25, 5, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("dtc_max_features","MAX FEATURES", 1, 14, 14, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("dtc_max_leaf_nodes","MAX LEAF NODES", 2, 50, 15, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("dtc_min_impurity_decrease","MAX IMPURITY DECREASE", 0, 1, 0, step=0.01, ticks=False)),
                        ui.div(div_parameters_style,ui.input_slider("dtc_min_samples_leaf","MIN SAMPLES LEAF", 1, 10, 1, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("dtc_min_samples_split","MIN SAMPLES SPLIT", 2, 10, 2, step=1, ticks=True)),
                    ),
                ),
                ui.nav("RFC",
                    ui.div(div_container_parameters_style,
                        ui.div(div_parameters_style,ui.input_slider("rfc_number_of_trees","NUMBER OF TREES", 1, 99, 11, step=2, ticks=True)),
                        ui.div(div_parameters_style,ui.input_select("rfc_criterion","CRITERION",["gini","entropy"])),
                        ui.div(div_parameters_style,ui.input_slider("rfc_max_depth","MAX DEPTH", 1, 25, 5, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("rfc_max_features","MAX FEATURES", 1, 14, 14, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("rfc_max_leaf_nodes","MAX LEAF NODES", 2, 50, 15, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("rfc_min_impurity_decrease","MAX IMPURITY DECREASE", 0, 1, 0, step=0.01, ticks=False)),
                        ui.div(div_parameters_style,ui.input_slider("rfc_min_samples_leaf","MIN SAMPLES LEAF", 1, 10, 1, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("rfc_min_samples_split","MIN SAMPLES SPLIT", 2, 10, 2, step=1, ticks=True)),
                    ),
                ),
                ui.nav("MLP",
                    ui.div(div_container_parameters_style,
                        ui.div(div_parameters_style,ui.input_slider("mlp_growth_rate","GROWTH RATE", 8, 36, 16, step=2, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("mlp_dropout","DROPOUT", 0, 1, 0.5, step=0.1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("mlp_epochs","EPOCHS", 100, 500, 250, step=10, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("mlp_learning_rate","LEARNING RATE", 0.001, 0.1, 0.01, step=0.001, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("mlp_batch_size","BATCH SIZE", 1, 32, 16, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("mlp_factor","FACTOR", 0, 1, 0.1, step=0.1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("mlp_patience","PATIENCE", 1, 20, 10, step=1, ticks=True)),
                        ui.div(div_parameters_style,ui.input_slider("mlp_epsilon","EPSILON", -10, -1, -8, step=1, ticks=True)),
                    ),
                ),
                ui.nav("_F_",
                    ui.div(div_container_global_parameters_style,
                        ui.div(div_container_parameters_style,
                            ui.div(div_parameters_style,ui.input_switch("abs_S_Smin","abs_S_Smin", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("rel_S_Smin_semi_width","rel_S_Smin_semi_width", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("rel_S_Smin_full_width","rel_S_Smin_full_width", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("abs_S_Smax","abs_S_Smax", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("rel_S_Smax_semi_width","rel_S_Smax_semi_width", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("rel_S_Smax_full_width","rel_S_Smax_full_width", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("count_anomalies_S","count_anomalies_S", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("ratio_anomalies_S","ratio_anomalies_S", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("max_variation_S","max_variation_S", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("mean_correlation","mean_correlation", value=True)),
                        ),
                        ui.div(div_container_parameters_style,
                            ui.div(div_parameters_style,ui.input_switch("abs_T_Tmin","abs_T_Tmin", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("rel_T_Tmin_semi_width","rel_T_Tmin_semi_width", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("rel_T_Tmin_full_width","rel_T_Tmin_full_width", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("abs_T_Tmax","abs_T_Tmax", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("rel_T_Tmax_semi_width","rel_T_Tmax_semi_width", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("rel_T_Tmax_full_width","rel_T_Tmax_full_width", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("count_anomalies_T","count_anomalies_T", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("ratio_anomalies_T","ratio_anomalies_T", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("max_variation_T","max_variation_T", value=True)),
                            ui.div(div_parameters_style,ui.input_switch("nb_measurements","nb_measurements", value=True)),
                        ),
                    ),
                ),
            ),
        ),
        ui.panel_main({"style": "height: 850px;"},
            ui.navset_tab(
                ui.nav("DECISION_TREE_CLASSIFIER",
                    ui.navset_tab(
                        ui.nav("TEST",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("dtc_launch_test", "LAUNCH")),
                                    ui.div(div_left_bottom_container_style,ui.output_text_verbatim("dtc_output_test_performances")),
                                ),
                                ui.div(div_right_container_style,ui.div(div_display_style,ui.output_image("dtc_output_test_tree"))),
                            ),
                        ),
                        ui.nav("CLASSIFICATION",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("dtc_launch_analysis_classification", "LAUNCH")),
                                ),
                                ui.div(div_right_container_style,ui.div(div_display_style,ui.output_text_verbatim("dtc_output_analysis_classification"))),
                            ),
                        ),
                        ui.nav("FEATURES",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("dtc_launch_analysis_features", "LAUNCH")),
                                ),
                                ui.div(div_right_container_style,ui.div(div_display_style,ui.output_image("dtc_output_analysis_features"))),
                            ),
                        ),
                    ),
                ),
                ui.nav("RANDOM_FOREST_CLASSIFIER",
                    ui.navset_tab(
                        ui.nav("TEST",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("rfc_launch_test", "LAUNCH")),
                                    ui.div(div_left_bottom_container_style,ui.output_text_verbatim("rfc_output_test_performances")),
                                ),
                            ),
                        ),
                        ui.nav("CLASSIFICATION",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("rfc_launch_analysis_classification", "LAUNCH")),
                                ),
                                ui.div(div_right_container_style,ui.div(div_display_style,ui.output_text_verbatim("rfc_output_analysis_classification"))),
                            ),
                        ),
                        ui.nav("FEATURES",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("rfc_launch_analysis_features", "LAUNCH")),
                                ),
                                ui.div(div_right_container_style,ui.div(div_display_style,ui.output_image("rfc_output_analysis_features"))),
                            ),
                        ),
                    ),
                ),
                ui.nav("MULTI_LAYER_PERCEPTRON",
                    ui.navset_tab(
                        ui.nav("TEST",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("mlp_launch_test", "LAUNCH")),
                                    ui.div(div_left_bottom_container_style,ui.output_text_verbatim("mlp_output_test_performances")),
                                ),
                                ui.div(div_right_container_style,ui.div(div_display_style,ui.output_image("mlp_output_test_loss"))),
                            ),
                        ),
                        ui.nav("CLASSIFICATION",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("mlp_launch_analysis_classification", "LAUNCH")),
                                ),
                                ui.div(div_right_container_style,ui.div(div_display_style,ui.output_text_verbatim("mlp_output_analysis_classification"))),
                            ),
                        ),
                        ui.nav("THRESHOLDS",
                            ui.div(div_main_container_style,
                                ui.div(div_left_container_style,
                                    ui.div(div_left_top_container_style,ui.input_action_button("mlp_launch_analysis_thresholds", "LAUNCH")),
                                    ui.div(div_left_bottom_container_style,ui.output_text_verbatim("mlp_output_analysis_thresholds_performances")),
                                ),
                                ui.div(div_right_container_style,ui.div(div_display_style,ui.output_image("mlp_output_analysis_thresholds"))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)


def server(input, output, session):

    MDS = reactive.Value("OFF")
    SDS = reactive.Value("OFF")

    @output
    @render.text
    def measures_dataset_status() :
        subprocess.run(['sh', '../launchers/sh/create_database.sh'], cwd=os.getcwd())
        return MDS.get()

    @output
    @render.text
    def stats_dataset_status() :
        subprocess.run(['sh', '../launchers/sh/create_database.sh'], cwd=os.getcwd())
        return SDS.get()
    
    @reactive.Effect
    @reactive.event(input.measures_dataset_create)
    def measures_dataset_create():
        if MDS.get() == "OFF" :
            subprocess.run(['sh', '../launchers/sh/delete_measures_dataset.sh'], cwd=os.getcwd())
            subprocess.run(['sh', '../launchers/sh/create_measures_dataset.sh'], cwd=os.getcwd())
            MDS.set("ON")

    @reactive.Effect
    @reactive.event(input.stats_dataset_create)
    def stats_dataset_create():
        if SDS.get() == "OFF" :
            subprocess.run(['sh', '../launchers/sh/delete_stats_dataset.sh'], cwd=os.getcwd())
            subprocess.run(['sh', '../launchers/sh/create_stats_dataset.sh'], cwd=os.getcwd())
            SDS.set("ON")

    @reactive.Effect
    @reactive.event(input.measures_dataset_delete)
    def measures_dataset_delete():
        if MDS.get() == "ON" :
            subprocess.run(['sh', '../launchers/sh/delete_measures_dataset.sh'], cwd=os.getcwd())
            MDS.set("OFF")

    @reactive.Effect
    @reactive.event(input.stats_dataset_delete)
    def stats_dataset_delete():
        if SDS.get() == "ON" :
            subprocess.run(['sh', '../launchers/sh/delete_stats_dataset.sh'], cwd=os.getcwd())
            SDS.set("OFF")

    @reactive.Effect
    @reactive.event(input.dtc_launch_test, input.dtc_launch_analysis_classification, input.dtc_launch_analysis_features)
    def update_dtc_parameters():
        configfile = "../conf/config.yaml"
        ymlfile = open(configfile, 'r')
        cfg = yaml.load(ymlfile, yaml.loader.SafeLoader)
        ymlfile.close()
        cfg["global_parameters"]["alarm_type"] = input.alarm_type()
        cfg["dataset"]["F_mult"] = input.fmult()
        cfg["dataset"]["T_mult"] = input.tmult()
        cfg["dtc_decision_tree_classifier"]["architecture"]["criterion"] = input.dtc_criterion()
        cfg["dtc_decision_tree_classifier"]["architecture"]["max_depth"] = input.dtc_max_depth()
        cfg["dtc_decision_tree_classifier"]["architecture"]["max_features"] = input.dtc_max_features()
        cfg["dtc_decision_tree_classifier"]["architecture"]["max_leaf_nodes"] = input.dtc_max_leaf_nodes()
        cfg["dtc_decision_tree_classifier"]["architecture"]["min_impurity_decrease"] = input.dtc_min_impurity_decrease()
        cfg["dtc_decision_tree_classifier"]["architecture"]["min_samples_leaf"] = input.dtc_min_samples_leaf()
        cfg["dtc_decision_tree_classifier"]["architecture"]["min_samples_split"] = input.dtc_min_samples_split()
        cfg["features"]["abs_S_Smin"] = input.abs_S_Smin()
        cfg["features"]["rel_S_Smin_semi_width"] = input.rel_S_Smin_semi_width()
        cfg["features"]["rel_S_Smin_full_width"] = input.rel_S_Smin_full_width()
        cfg["features"]["abs_S_Smax"] = input.abs_S_Smax()
        cfg["features"]["rel_S_Smax_semi_width"] = input.rel_S_Smax_semi_width()
        cfg["features"]["rel_S_Smax_full_width"] = input.rel_S_Smax_full_width()
        cfg["features"]["count_anomalies_S"] = input.count_anomalies_S()
        cfg["features"]["ratio_anomalies_S"] = input.ratio_anomalies_S()
        cfg["features"]["max_variation_S"] = input.max_variation_S()
        cfg["features"]["abs_T_Tmin"] = input.abs_T_Tmin()
        cfg["features"]["rel_T_Tmin_semi_width"] = input.rel_T_Tmin_semi_width()
        cfg["features"]["rel_T_Tmin_full_width"] = input.rel_T_Tmin_full_width()
        cfg["features"]["abs_T_Tmax"] = input.abs_T_Tmax()
        cfg["features"]["rel_T_Tmax_semi_width"] = input.rel_T_Tmax_semi_width()
        cfg["features"]["rel_T_Tmax_full_width"] = input.rel_T_Tmax_full_width()
        cfg["features"]["count_anomalies_T"] = input.count_anomalies_T()
        cfg["features"]["ratio_anomalies_T"] = input.ratio_anomalies_T()
        cfg["features"]["max_variation_T"] = input.max_variation_T()
        cfg["features"]["mean_correlation"] = input.mean_correlation()
        cfg["features"]["nb_measurements"] = input.nb_measurements()
        ymlfile2 = open(configfile, 'w')
        yaml.dump(cfg, ymlfile2)
        ymlfile2.close()

    @reactive.Effect
    @reactive.event(input.rfc_launch_test, input.rfc_launch_analysis_classification, input.rfc_launch_analysis_features)
    def update_rfc_parameters():
        configfile = "../conf/config.yaml"
        ymlfile = open(configfile, 'r')
        cfg = yaml.load(ymlfile, yaml.loader.SafeLoader)
        ymlfile.close()
        cfg["global_parameters"]["alarm_type"] = input.alarm_type()
        cfg["dataset"]["F_mult"] = input.fmult()
        cfg["dataset"]["T_mult"] = input.tmult()
        cfg["rfc_random_forest_classifier"]["architecture"]["criterion"] = input.rfc_criterion()
        cfg["rfc_random_forest_classifier"]["architecture"]["max_depth"] = input.rfc_max_depth()
        cfg["rfc_random_forest_classifier"]["architecture"]["max_features"] = input.rfc_max_features()
        cfg["rfc_random_forest_classifier"]["architecture"]["max_leaf_nodes"] = input.rfc_max_leaf_nodes()
        cfg["rfc_random_forest_classifier"]["architecture"]["min_impurity_decrease"] = input.rfc_min_impurity_decrease()
        cfg["rfc_random_forest_classifier"]["architecture"]["min_samples_leaf"] = input.rfc_min_samples_leaf()
        cfg["rfc_random_forest_classifier"]["architecture"]["min_samples_split"] = input.rfc_min_samples_split()
        cfg["rfc_random_forest_classifier"]["architecture"]["number_of_trees"] = input.rfc_number_of_trees()
        cfg["features"]["abs_S_Smin"] = input.abs_S_Smin()
        cfg["features"]["rel_S_Smin_semi_width"] = input.rel_S_Smin_semi_width()
        cfg["features"]["rel_S_Smin_full_width"] = input.rel_S_Smin_full_width()
        cfg["features"]["abs_S_Smax"] = input.abs_S_Smax()
        cfg["features"]["rel_S_Smax_semi_width"] = input.rel_S_Smax_semi_width()
        cfg["features"]["rel_S_Smax_full_width"] = input.rel_S_Smax_full_width()
        cfg["features"]["count_anomalies_S"] = input.count_anomalies_S()
        cfg["features"]["ratio_anomalies_S"] = input.ratio_anomalies_S()
        cfg["features"]["max_variation_S"] = input.max_variation_S()
        cfg["features"]["abs_T_Tmin"] = input.abs_T_Tmin()
        cfg["features"]["rel_T_Tmin_semi_width"] = input.rel_T_Tmin_semi_width()
        cfg["features"]["rel_T_Tmin_full_width"] = input.rel_T_Tmin_full_width()
        cfg["features"]["abs_T_Tmax"] = input.abs_T_Tmax()
        cfg["features"]["rel_T_Tmax_semi_width"] = input.rel_T_Tmax_semi_width()
        cfg["features"]["rel_T_Tmax_full_width"] = input.rel_T_Tmax_full_width()
        cfg["features"]["count_anomalies_T"] = input.count_anomalies_T()
        cfg["features"]["ratio_anomalies_T"] = input.ratio_anomalies_T()
        cfg["features"]["max_variation_T"] = input.max_variation_T()
        cfg["features"]["mean_correlation"] = input.mean_correlation()
        cfg["features"]["nb_measurements"] = input.nb_measurements()
        ymlfile2 = open(configfile, 'w')
        yaml.dump(cfg, ymlfile2)
        ymlfile2.close()

    @reactive.Effect
    @reactive.event(input.mlp_launch_test, input.mlp_launch_analysis_classification, input.mlp_launch_analysis_thresholds)
    def update_mlp_parameters():
        configfile = "../conf/config.yaml"
        ymlfile = open(configfile, 'r')
        cfg = yaml.load(ymlfile, yaml.loader.SafeLoader)
        ymlfile.close()
        cfg["global_parameters"]["alarm_type"] = input.alarm_type()
        cfg["dataset"]["F_mult"] = input.fmult()
        cfg["dataset"]["T_mult"] = input.tmult()
        cfg["mlp_multi_layer_perceptron"]["batch_size"] = input.mlp_batch_size()
        cfg["mlp_multi_layer_perceptron"]["dropout"] = input.mlp_dropout()
        cfg["mlp_multi_layer_perceptron"]["epochs"] = input.mlp_epochs()
        cfg["mlp_multi_layer_perceptron"]["epsilon"] = 10**(input.mlp_epsilon())
        cfg["mlp_multi_layer_perceptron"]["factor"] = input.mlp_factor()
        cfg["mlp_multi_layer_perceptron"]["growth_rate"] = input.mlp_growth_rate()
        cfg["mlp_multi_layer_perceptron"]["learning_rate"] = input.mlp_learning_rate()
        cfg["mlp_multi_layer_perceptron"]["patience"] = input.mlp_patience()
        cfg["features"]["abs_S_Smin"] = input.abs_S_Smin()
        cfg["features"]["rel_S_Smin_semi_width"] = input.rel_S_Smin_semi_width()
        cfg["features"]["rel_S_Smin_full_width"] = input.rel_S_Smin_full_width()
        cfg["features"]["abs_S_Smax"] = input.abs_S_Smax()
        cfg["features"]["rel_S_Smax_semi_width"] = input.rel_S_Smax_semi_width()
        cfg["features"]["rel_S_Smax_full_width"] = input.rel_S_Smax_full_width()
        cfg["features"]["count_anomalies_S"] = input.count_anomalies_S()
        cfg["features"]["ratio_anomalies_S"] = input.ratio_anomalies_S()
        cfg["features"]["max_variation_S"] = input.max_variation_S()
        cfg["features"]["abs_T_Tmin"] = input.abs_T_Tmin()
        cfg["features"]["rel_T_Tmin_semi_width"] = input.rel_T_Tmin_semi_width()
        cfg["features"]["rel_T_Tmin_full_width"] = input.rel_T_Tmin_full_width()
        cfg["features"]["abs_T_Tmax"] = input.abs_T_Tmax()
        cfg["features"]["rel_T_Tmax_semi_width"] = input.rel_T_Tmax_semi_width()
        cfg["features"]["rel_T_Tmax_full_width"] = input.rel_T_Tmax_full_width()
        cfg["features"]["count_anomalies_T"] = input.count_anomalies_T()
        cfg["features"]["ratio_anomalies_T"] = input.ratio_anomalies_T()
        cfg["features"]["max_variation_T"] = input.max_variation_T()
        cfg["features"]["mean_correlation"] = input.mean_correlation()
        cfg["features"]["nb_measurements"] = input.nb_measurements()
        ymlfile2 = open(configfile, 'w')
        yaml.dump(cfg, ymlfile2)
        ymlfile2.close()

    @reactive.Effect
    @reactive.event(input.dtc_launch_test)
    def dtc_launch_test():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/dtc_test.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.text
    @reactive.event(input.dtc_launch_test)
    def dtc_output_test_performances() :
        if SDS.get() == "ON" :
            file = open("../data/dtc_decision_tree_classifier/performances.txt")
            text = file.read()
            file.close()
            return text
    
    @output
    @render.image
    @reactive.event(input.dtc_launch_test)
    def dtc_output_test_tree() :
        if SDS.get() == "ON" :
            img: ImgData = {"src": ("../data/dtc_decision_tree_classifier/tree.png"), "height":"600px;"}
            return img

    @reactive.Effect
    @reactive.event(input.dtc_launch_analysis_classification)
    def dtc_launch_analysis_classification():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/dtc_analysis_classification.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.text
    @reactive.event(input.dtc_launch_analysis_classification)
    def dtc_output_analysis_classification() :
        if SDS.get() == "ON" :
            file = open("../data/dtc_decision_tree_classifier/classification.txt")
            text = file.read()
            file.close()
            return text

    @reactive.Effect
    @reactive.event(input.dtc_launch_analysis_features)
    def dtc_launch_analysis_features():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/dtc_analysis_features.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.image
    @reactive.event(input.dtc_launch_analysis_features)
    def dtc_output_analysis_features() :
        if SDS.get() == "ON" :
            img: ImgData = {"src": ("../data/dtc_decision_tree_classifier/features.png"), "height":"600px;"}
            return img
    
    @reactive.Effect
    @reactive.event(input.rfc_launch_test)
    def rfc_launch_test():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/rfc_test.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.text
    @reactive.event(input.rfc_launch_test)
    def rfc_output_test_performances() :
        if SDS.get() == "ON" :
            file = open("../data/rfc_random_forest_classifier/performances.txt")
            text = file.read()
            file.close()
            return text

    @reactive.Effect
    @reactive.event(input.rfc_launch_analysis_classification)
    def rfc_launch_analysis_classification():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/rfc_analysis_classification.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.text
    @reactive.event(input.rfc_launch_analysis_classification)
    def rfc_output_analysis_classification() :
        if SDS.get() == "ON" :
            file = open("../data/rfc_random_forest_classifier/classification.txt")
            text = file.read()
            file.close()
            return text

    @reactive.Effect
    @reactive.event(input.rfc_launch_analysis_features)
    def rfc_launch_analysis_features():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/rfc_analysis_features.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.image
    @reactive.event(input.rfc_launch_analysis_features)
    def rfc_output_analysis_features() :
        if SDS.get() == "ON" :
            img: ImgData = {"src": ("../data/rfc_random_forest_classifier/features.png"), "height":"600px"}
            return img

    @reactive.Effect
    @reactive.event(input.mlp_launch_test)
    def mlp_launch_test():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/mlp_test.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.text
    @reactive.event(input.mlp_launch_test)
    def mlp_output_test_performances() :
        if SDS.get() == "ON" :
            file = open("../data/mlp_multi_layer_perceptron/performances.txt")
            text = file.read()
            file.close()
            return text
    
    @output
    @render.image
    @reactive.event(input.mlp_launch_test)
    def mlp_output_test_loss() :
        if SDS.get() == "ON" :
            img: ImgData = {"src": ("../data/mlp_multi_layer_perceptron/mlp_loss.png"), "height":"600px;"}
            return img

    @reactive.Effect
    @reactive.event(input.mlp_launch_analysis_classification)
    def mlp_launch_analysis_classification():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/mlp_analysis_classification.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.text
    @reactive.event(input.mlp_launch_analysis_classification)
    def mlp_output_analysis_classification() :
        if SDS.get() == "ON" :
            file = open("../data/mlp_multi_layer_perceptron/classification.txt")
            text = file.read()
            file.close()
            return text

    @reactive.Effect
    @reactive.event(input.mlp_launch_analysis_thresholds)
    def mlp_launch_analysis_thresholds():
        if SDS.get() == "ON" : subprocess.run(['sh', '../launchers/sh/mlp_analysis_thresholds.sh'], cwd=os.getcwd())
        else : ui.notification_show("Stats Dataset is not ready", duration=5)

    @output
    @render.text
    @reactive.event(input.mlp_launch_analysis_thresholds)
    def mlp_output_analysis_thresholds_performances() :
        if SDS.get() == "ON" :
            file = open("../data/mlp_multi_layer_perceptron/performances.txt")
            text = file.read()
            file.close()
            return text
    
    @output
    @render.image
    @reactive.event(input.mlp_launch_analysis_thresholds)
    def mlp_output_analysis_thresholds() :
        if SDS.get() == "ON" :
            img: ImgData = {"src": ("../data/mlp_multi_layer_perceptron/mlp_thresholds.png"), "height":"600px;"}
            return img

app = App(app_ui, server, debug=True)