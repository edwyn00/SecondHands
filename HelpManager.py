"""
Created by Edoardo Alati
"""

from CONFIG     import *
from LIBRARIES  import *

class HelpManager:

    def __init__(self):


        '''
        Context dictionary via CONFIG values replaces the args parser of the previous version
        '''

        self.context = {}
        self.context['width']                           = WIDTH
        self.context['height']                          = HEIGHT
        self.context['send_command']                    = SEND_COMMAND
        self.context['framerate']                       = FRAMERATE
        self.context['ice_publishing_address']          = ICE_PUBLISHING_ADDRESS
        self.context['ice_publishing_port']             = ICE_PUBLISHING_PORT
        self.context['ice_subscribing_address_table']   = ICE_SUBSCRIBING_ADDRESS_TABLE
        self.context['ice_subscribing_address_scene']   = ICE_SUBSCRIBING_ADDRESS_SCENE
        self.context['image_type']                      = IMAGE_TYPE
        self.context['video_file']                      = VIDEO_FILE
        self.context['threads']                         = THREADS
        self.context['n_clips']                         = N_CLIPS
        self.context['n_clip_frames']                   = N_CLIP_FRAMES
        self.context['n_skipped_frames']                = N_SKIPPED_FRAMES
        self.context['time_window']                     = TIME_WINDOW
        self.context['estimation_step_time']            = ESTIMATION_STEP_TIME
        self.context['verbose']                         = VERBOSE
        self.context['display']                         = DISPLAY
        self.context['device_help']                     = DEVICE_HELP
        self.context['device_maskrcnn']                 = DEVICE_MASKRCNN

        '''
        Communication events list:
        INPUTS: Multiple Cameras, Speech System, State Charts
        OUTPUTS: Speech System, State Charts, Mask Results
        '''

        self.input_output_manager = None #Safe Declaration
        input_output_init()



    '''Can we use the CONFIG.py file?'''

    def define_consts(self):
        global DEVICE_LIST, FILES_PATH, META_GRAPH_PATH, CHECKPOINT_PATH_HELP,  LABELS_FILE, \
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
        DEVICE_LIST = ("/cpu:0","/gpu:0","/gpu:1","/gpu:2")
        CHECKPOINT_PATH_MASK = os.path.join(PROJECT_DIR, 'Dependencies', 'Help-System',
                                        'Data', 'model_data', 'tf_mask_rcnn')

        # # various
        SCALE_FACTOR = 1.0
        NO_SKIPPING_FACTOR = 0
        PROCESSING_WIDTH = 640
        PROCESSING_HEIGHT = 480

    def help_converter(self, our_help):
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

    def mask_converter(self, dobj, clist):
        not_supported = ['location', 'technician', 'guard-support', 'diverter']
        dout = {}
        for obj in dobj:
            if obj not in not_supported:
                dout[clist[obj]] = dobj[obj]
        return dout

    def network_config(self): 
        # activity labels
        with open(LABELS_FILE, 'rb') as f:
            labels = pickle.load(f)
        with open(LABELS_FILE_INV, 'rb') as f:
            labels_inv = pickle.load(f)
        # tensorflow session configuration
        #tf_config = None
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=self.context['threads'])
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement=True
        tf_config.log_device_placement=False

        return labels, labels_inv, tf_config

    def main(self):
        # initialisations =========================================================
        define_consts()
        labels, labels_inv, tf_config = network_config()

        img_width = PROCESSING_WIDTH
        img_height = PROCESSING_HEIGHT

        # application blocks building =============================================

        '''
        The image server & packer will be initialized here
        '''

        is_recognising = True

        # install SIGINT handler
        def signal_handler(sig, frame):
            nonlocal is_recognising
            print('You pressed Ctrl+C, gracefully quitting application...')
            is_recognising = False
        signal.signal(signal.SIGINT, signal_handler)

        #####################
        #   MASK RCNN INIT  #
        #####################


        mask_ready = threading.Event()
        graph_mask = tf.Graph()
        sess_mask = tf.Session(config=tf_config, graph=graph_mask)
        mask_network = MaskWrapper(checkpoint_path=CHECKPOINT_PATH_MASK, sess=sess_mask, graph=None,
                                        device=self.context['device_maskrcnn'], display=self.context['display'])


        #####################
        #   HELP NET INIT   #
        #####################

        '''
        graph_help = tf.Graph()
        sess_help = tf.Session(config=tf_config, graph=graph_help)
        help_network = HelpWrapper(META_GRAPH_PATH ,CHECKPOINT_PATH_HELP, sess=sess_help,
                                        graph=None, device=self.context['device_help'], labels=labels, display=self.context['display'])
        help_network.set_images_cache(NewImagesCaching(help_network.nn, images_packer_table.get_shape()))
        '''


        print("Starting activity recognition loop, press Ctrl+C to quit.")


        '''
        The Sarch Action will be performed here
        '''



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


        # main loop ===============================================================

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
            if  self.context['send_commands'] and 'sil' not in help_pred_label[0]:
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

    def input_output_init(self):
        self.input_cameras_event        = threading.Event()
        self.input_speech_system_event  = threading.Event()
        self.input_state_charts_event   = threading.Event()
        self.output_mask_event          = threading.Event()
        self.output_speech_system_event = threading.Event()
        self.output_state_charts_event  = threading.Event()

        input_output_events = [ self.input_cameras_event,           \
                                self.input_speech_system_event,     \
                                self.input_state_charts_event,      \
                                self.output_mask_event,             \
                                self.output_speech_system_event,    \
                                self.output_state_charts_event      ]

        self.input_output_manager = InputOutputManager(input_output_events)
