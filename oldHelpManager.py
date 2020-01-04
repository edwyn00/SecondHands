#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Fiora Pirri, Valsamis Ntouskos, Edoardo Alati, Lorenzo Mauro
"""
import __init__
from __init__ import PROJECT_DIR

from activity_recognition import StreamImagesPacker, VideoImagesPacker, \
                                NewImagesCaching

import activity_recognition.armarx_helper as armarx_helper
from activity_recognition.armarx_helper import StatechartExecutorInterfacePrx, \
                                    PlanStepVariantBase, EntityRefBase

from tf_mask_rcnn.mask_wrapper import MaskWrapper
from help_recognition.help_wrapper import HelpWrapper

from shared_information_sources.measurement_manager import \
    cv2ImageFromStringManager, cv2ImageFromVideoManager
from ice_communication import MultiTopicPublisherComposite, \
    PublisherComposite, IceTextPublisher, ImageBroadcasterServant, \
    ice_configuration
from ice_management import Manager
from ImageBroadcasting import ColourSpace
from ImageServerRequesting import (
    ImageColorType,
    ImageContent,
    ImageServerRequestPrx,
    RequestInfo,
)
import Texting

import tensorflow as tf
import os
import sys
import argparse
import time
import pickle
import cv2
import signal
import numpy as np

from PIL import Image

import Ice
import threading

DEVICE_LIST = ("/cpu:0","/gpu:0","/gpu:1","/gpu:2")

def define_consts():
    global FILES_PATH, META_GRAPH_PATH, CHECKPOINT_PATH_HELP,  LABELS_FILE, \
        PROCESSING_WIDTH, PROCESSING_HEIGHT, ARGPARSE_CONFIG, SCALE_FACTOR, \
        NO_SKIPPING_FACTOR, CHECKPOINT_PATH_MASK, LABELS_FILE_INV
    # constants initialisation
    # # file paths
    FILES_PATH = os.path.join(PROJECT_DIR, 'Data', 'video_test')
    MODEL_PATH = os.path.join(PROJECT_DIR, 'Dependencies', 'Help-System',
                                    'Data', 'model_data', 'help_network')
    META_GRAPH_PATH = os.path.join(MODEL_PATH, 'activity_network_model.ckpt.meta')
    CHECKPOINT_PATH_HELP = MODEL_PATH
    LABELS_FILE = os.path.join(MODEL_PATH, 'id_to_word.pkl')
    LABELS_FILE_INV = os.path.join(MODEL_PATH, 'word_to_id.pkl')

    CHECKPOINT_PATH_MASK = os.path.join(PROJECT_DIR, 'Dependencies', 'Help-System',
                                    'Data', 'model_data', 'tf_mask_rcnn')

    # # various
    SCALE_FACTOR = 1.0
    NO_SKIPPING_FACTOR = 0
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 480

    # # cli flags
    ARGPARSE_CONFIG = {
        '--width': {  # currently unsupported
            'type': int,
            'help': 'width resolution of the streamed images (fixed)',
            'default': PROCESSING_WIDTH
        },
        '--height': {  # currently unsupported
            'type': int,
            'help': 'height resolution of the streamed images (fixed)',
            'default': PROCESSING_HEIGHT
        },
        '--send-commands': {
            'help': 'send commands to robot',
            'action': 'store_true',
            'default': False
        },
        '--framerate': {
            'type': int,
            'help': 'refresh rate of the streamed images',
            'default': 30,
        },
        '--ice-publishing-address': {
            'help': 'IP address of the Ice publishingservices provider',
            'default': '',
        },
        '--ice-publishing-port': {
            'help': 'TCP port associated to the Ice publishing services \
                provider',
            'default': '4061',
        },
        '--ice-subscribing-address-table': {
            'help': 'IP address of the Ice subscribing services provider',
            'default': '',
        },
        '--ice-subscribing-port-table': {
            'help': 'TCP port associated to the Ice subscribing services \
                provider',
            'default': '4061',
        },
        '--ice-subscribing-address-scene': {
            'help': 'IP address of the Ice subscribing services provider',
            'default': '',
        },
        '--ice-subscribing-port-scene': {
            'help': 'TCP port associated to the Ice subscribing services \
                provider',
            'default': '4061',
        },
        '--image-type': {
            'help': 'type of the images retrieved',
            'default': 'imagery'
        },
        '--video-file': {
            'help': 'video file name to fetch from Data/video_test (no path)',
            'default': '',
        },
        '--threads': {
            'type': int,
            'help': 'threading capability of the used machine',
            'default': 7,
        },
        '--n-clips': {
            'type': int,
            'help': 'number of processing clips',
            'default': 4,
        },
        '--n-clip-frames': {
            'type': int,
            'help': 'number of frames for clip',
            'default': 10,
        },
        '--n-skipped-frames': {
            'type': float,
            'help': 'frames skipped within the clip (between two valid \
                frames)',
            'default': 1.,
        },
        '--time-window': {
            'type': float,
            'help': 'length of the time-window for a single assessment \
                iteration, secs',
            'default': 2.6,
        },
        '--estimation-step-time': {
            'type': float,
            'help': 'time window between two different estimation steps, secs',
            'default': 1,
        },
        '--verbose': {
            'help': 'verbose output',
            'action': 'store_true',
            'default': False
        },
        '--display': {
            'help': 'displays captured images overlaid with the label \
                recognised by the network',
            'action': 'store_true',
            'default': False
        },
        '--device-help': {  # currently unsupported
            'type': str,
            'help': 'set device for help recognition network',
            'default': "/gpu:0",
            'choices': DEVICE_LIST
        },
        '--device-maskrcnn': {  # currently unsupported
            'type': str,
            'help': 'set device for Mask RCNN network',
            'default': "/gpu:0",
            'default': DEVICE_LIST
        },
    }


def check_parameter_assertions(parameters):
    assert (parameters['video_file'] or ((
        not parameters['video_file']) and
        parameters['ice_subscribing_address_scene'] or parameters['ice_subscribing_address_table'])),\
        "Required address of Ice grid for image subscribing"


def parse_args():
    """ Parse the program arguments into a dictionary. """
    ap = argparse.ArgumentParser(description=__doc__)

    for name, entry in ARGPARSE_CONFIG.items():
        ap.add_argument(name, **entry)

    return ap.parse_args(sys.argv[1:]).__dict__.copy()

def send_command(proxy, help, args):
    class EntityRef(EntityRefBase):
        pass

    class PlanStepVariant(PlanStepVariantBase):
        pass

    executor = StatechartExecutorInterfacePrx.checkedCast(proxy)
    planstep = PlanStepVariant()
    planstep.name = help

    planargs = []
    for i in range(len(args)):
        entity = EntityRef()
        entity.entityName = args[i]
        planargs.append(entity)

    planstep.args = planargs

    query = ''
    query = str(executor.ice_invoke("getCurrentState", Ice.OperationMode.Normal, bytes())[1])

    if 'running' not in query:
        print("Sending command!")
        result = executor.execute(planstep) #begin_execute() for async / executor.end_execute(async)
        print("Command finished!")
        print(str(result))
    else:
        print("Armar is busy")


def help_converter(our_help):
    '''Convert from our representation to armar's

    Example
    -------
    input = ['get_from_technician_and_put_on_the_table', 'spray_bottle', 'on_ladder']
    output = ('PutObjectAway', ['spraybottle', 'handover_ladder', 'wb_front'])
    '''
    armar_location = {'under_diverter': 'handover_under_conveyor', 'on_ladder': 'handover_ladder' ,
                    'guard': 'handover_ladder', 'at_guard_support': 'handover_ladder'}
    armar_object ={'torch' : 'torch', 'cloth' : 'cloth', 'spray_bottle': 'spraybottle',
                    'cutter': 'cutter', 'guard': ''}

    armar_help = None
    action = our_help[0]
    if action == 'remove_and_put_down':
        armar_help = 'GuardSupport', []
    elif action == 'grasp_and_put_on_diverter':
        armar_help = 'LiftGuard', []
    else:
        obj = our_help[1]
        loc = our_help[2]
        if action == 'give_to_technician':
            armar_help = 'BringObject', [armar_object[obj], 'wb_front', armar_location[loc]]
        else:
            armar_help = 'PutObjectAway', [armar_object[obj], armar_location[loc], 'wb_front']
    return armar_help

def mask_converter(dobj, clist):
    not_supported = ['location', 'technician', 'guard-support', 'diverter']
    dout = {}
    for obj in dobj:
        if obj not in not_supported:
            dout[clist[obj]] = dobj[obj]
    return dout

def main():
    # initialisations =========================================================
    define_consts()
    context = parse_args()
    check_parameter_assertions(context)
    labels, labels_inv, tf_config = network_config(context)

    img_width = PROCESSING_WIDTH  # TEMP
    img_height = PROCESSING_HEIGHT  # TEMP  # at present, only 368x368 is legal

    # application blocks building =============================================

    if (context['video_file']):
        print("Running on video input")
        # images from video
        image_manager, packer_class = get_from_video(context)
    else:
        # otherwise image from stream
        subscribing_comms_manager_table = build_ice_manager(context, 'subscribing', 'table')
        image_manager_table, packer_class_table = get_from_stream(context,
                                                      subscribing_comms_manager_table
                                                     )

        subscribing_comms_manager_scene = build_ice_manager(context, 'subscribing', 'scene')
        image_manager_scene, packer_class_scene = get_from_stream(context,
                                                      subscribing_comms_manager_scene
                                                     )

    iceCommunicator = armarx_helper.get_communicator()
    statechart_proxy = iceCommunicator.stringToProxy('StatechartExecutorObserver')

    if context['ice_publishing_address']:
        publishing_comms_manager = build_ice_manager(context, 'publishing')
        output_proxy = setup_proxy(publishing_comms_manager.
                                   createProxyUsingTopic,
                                   "ActivityRecognitionLabel",
                                   Texting.TextMessengerPrx)
    else:
        output_proxy = None

    # images-packer instantiation
    images_packer_table = packer_class_table(
        image_manager_table,
        img_width,
        img_height,
        context['n_clips'],
        context['n_clip_frames'],
        context['time_window'] * context['framerate'],
        context['estimation_step_time'] * context['framerate'],
        context['n_skipped_frames'])

    images_packer_scene = packer_class_scene(
        image_manager_scene,
        img_width,
        img_height,
        context['n_clips'],
        context['n_clip_frames'],
        context['time_window'] * context['framerate'],
        context['estimation_step_time'] * context['framerate'],
        context['n_skipped_frames'])
    # main loop ===============================================================
    is_recognising = True

    # install SIGINT handler
    def signal_handler(sig, frame):
        nonlocal is_recognising
        print('You pressed Ctrl+C, gracefully quitting application...')
        is_recognising = False
    signal.signal(signal.SIGINT, signal_handler)

    # mask_ready = threading.Event()

    graph_mask = tf.Graph()
    sess_mask = tf.Session(config=tf_config, graph=graph_mask)
    mask_network = MaskWrapper(checkpoint_path=CHECKPOINT_PATH_MASK, sess=sess_mask, graph=None,
                                    device=context['device_maskrcnn'], display=context['display'])

    graph_help = tf.Graph()
    sess_help = tf.Session(config=tf_config, graph=graph_help)
    help_network = HelpWrapper(META_GRAPH_PATH ,CHECKPOINT_PATH_HELP, sess=sess_help,
                                    graph=None, device=context['device_help'], labels=labels, display=context['display'])


    help_network.set_images_cache(NewImagesCaching(help_network.nn, images_packer_table.get_shape()))

    print("Starting activity recognition loop, press Ctrl+C to quit.")
    # print(labels)

    images_pack_scene=None
    images_pack_table=None

    while images_pack_scene is None or images_pack_table is None:
        print("Not enough frames to perform computation")
        time.sleep(1)
        images_pack_scene, images_list_scene = images_packer_scene.get_images_pack()
        images_pack_table, images_list_table = images_packer_table.get_images_pack()
        continue

    object_seconds = []
    image_groups = [[images_pack_table[0, i, 0, ...], images_pack_scene[0, i, 0, ...]] for i in range(4)]
    for group in image_groups:
        images = [im[:,:,::-1] for im in group]
        images = [np.array(Image.fromarray(im).resize((int(1080*4/3),1080), resample=Image.BICUBIC)) for im in images]
        mask_network.set_data(images)
        t_mask = mask_network.spin()
        t_mask.join()
        mask_output, mask_results = mask_network.get_data()
        obj_help = mask_converter(mask_output, labels_inv)
        object_seconds.append(obj_help)


    while is_recognising:

        images_pack_scene, images_list_scene = images_packer_scene.get_images_pack()

        second = int(images_list_scene[-1]/images_packer_scene._clip_distance) - 1
        print("Second", second)


        help_network.set_data(images_pack_scene, images_list_scene, second, object_seconds)

        t_help = help_network.spin()

        images_pack_table, images_list_table = images_packer_table.get_images_pack()

        images = [images_pack_table[0, -1, 0, ...], images_pack_scene[0, -1, 0, ...]]
        images = [im[:,:,::-1] for im in images]
        images = [np.array(Image.fromarray(im).resize((int(1080*4/3),1080), resample=Image.BICUBIC)) for im in images]

        mask_network.set_data(images)
        t_mask = mask_network.spin()

        t_help.join()
        t_mask.join()


        help_sm, now_sm, _ = help_network.get_data()
        help_pred_label, y_pred_label, help_pred, y_pred = help_network.get_predictions()
        print("Actions:", y_pred_label, np.max(now_sm, axis=-1))

        print("Help:", help_pred_label)
        if  context['send_commands'] and 'sil' not in help_pred_label[0]:
            armar_help = help_converter(help_pred_label)
            print(armar_help)
            send_command(statechart_proxy, armar_help[0], armar_help[1])

        mask_output, mask_results = mask_network.get_data()
        object_seconds.pop(0)
        obj_help = mask_converter(mask_output, labels_inv)
        print([labels[obj_id] for obj_id in obj_help])
        object_seconds.append(obj_help)

        help_network.visualize(int(1000/6))
        mask_network.visualize()



# Main function end ***********************************************************


def network_config(context):
    # activity labels
    with open(LABELS_FILE, 'rb') as f:
        labels = pickle.load(f)
    with open(LABELS_FILE_INV, 'rb') as f:
        labels_inv = pickle.load(f)
    # tensorflow session configuration
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=context['threads'])
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement=True
    tf_config.log_device_placement=False

    return labels, labels_inv, tf_config


def publish_string(output_proxy, label):
    text_message = Texting.TextMessage()
    text_message.message = label
    output_proxy.onMessage(text_message)


def displayResult(images_pack, label):

    # TODO the below fors can be vectorised for improved performance
    # TODO remove magic numbers
    for id in range(images_pack.shape[1]): #range(images_pack.shape[1]):
        for fr in range(images_pack.shape[2]):
                frame = images_pack[0, id, fr, :, :, :]
                # write white label, 1/2 of original font size, top left corner
                # # of image, standard thickness (1)
                cv2.putText(frame, label, (round(frame.shape[0] * 0.05),
                                           round(frame.shape[1] * 0.05)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0xFF, 0xFF, 0xFF), 1)
                cv2.imshow('Recognition results', frame)




def setup_request(context):
    request = RequestInfo()
    request.content = ImageContent.depth if \
        context['image_type'] == "depth" else ImageContent.imagery
    request.colorType = ImageColorType.eRgb
    request.framerate = context['framerate']
    request.imageWidth = PROCESSING_WIDTH  # TEMP
    request.imageHeight = PROCESSING_HEIGHT  # TEMP
    return request


def setup_proxy(f_create_comms, topic_name, topic_if):
    proxy_params = Manager.proxyParams()
    proxy_params['communicationDirection'] = Manager.CommunicationDirection.\
        TwoWay
    proxy_params['toggleCheck'] = Manager.ToggleCheck.NoCheck
    publishing_proxy = f_create_comms(topic_if, "%s" % topic_name,
                                      proxy_params)
    return publishing_proxy


def build_ice_manager(context, role, nr):
    config = ice_configuration(
        {'ice_address': context['ice_{}_address_{}'.format(role, nr)],
         'ice_port': context['ice_{}_port_{}'.format(role, nr)]}
    )
    manager = Manager.Builder().useConfig(config).build()

    return manager


def create_servant(image_manager, manager, reply):
    servantParams = Manager.servantParams()
    servantParams['usingAdmin'] = False
    servantParams['communicationDirection'] = Manager.CommunicationDirection.\
        OneWay
    manager.createServantUsingTopic(
        ImageBroadcasterServant,
        reply.topic,
        "ImageBroadcasterAdapter",
        servantParams,
        [image_manager]
    )


def get_from_stream(context, manager):
    image_manager = cv2ImageFromStringManager({
        'scale': SCALE_FACTOR,
        'colour_space': ColourSpace.BGR,
        'source_colour_space': ColourSpace.BGR,
        'buffering': True,
        'buffer_capacity': 2 * context['time_window'] * context['framerate']
    })

    # request proxy
    publishing_proxy = setup_proxy(manager.createProxyUsingId, "ImageServer",
                                   ImageServerRequestPrx)
    request = setup_request(context)
    reply = publishing_proxy.serviceRequest(request)
    print(reply)

    # servant for image frames
    create_servant(image_manager, manager, reply)

    return image_manager, StreamImagesPacker


def get_from_video(context):
    # source setup
    video_manager = cv2ImageFromVideoManager()
    video_manager.use_file(os.path.join(FILES_PATH, context['video_file']),
                           {'scale': SCALE_FACTOR,
                            'skipping_factor': NO_SKIPPING_FACTOR,
                            'colour_space': ColourSpace.BGR,
                            'source_colour_space': ColourSpace.BGR,
                            'height': PROCESSING_HEIGHT,
                            'width': PROCESSING_WIDTH})
    video_manager.update()

    return video_manager, VideoImagesPacker


if __name__ == '__main__':
    sys.exit(main())
