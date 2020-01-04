from LIBRARIES          import *
from CONFIG             import *

class InputOutputManager(threading.Thread):

    '''
    Output events (state charts and speech system) will propagate to the reasoning thread and will pause the process,
    until an boolean value (for state charts) or a json (for the speech system) is received

    Input events are used to notice to the help thread that new information is available.
    E.g. new frames, technician override commands or robot pose during search
    '''

    def __init__(self, events_list):
        threading.Thread.__init__(self)

        '''
        self.input_cameras_event        = events_list[0]
        self.input_speech_system_event  = events_list[1]
        self.input_state_charts_event   = events_list[2]
        self.output_mask_event          = events_list[3]
        self.output_speech_system_event = events_list[4]
        self.output_state_charts_event  = events_list[5]
        '''

        self.events_list = events_list

    def run(self):

        while(True):
            while any(singleEvent.isSet() for singleEvent in self.events_list):

                #input_cameras_event
                if events_list[0].isSet():
                    events_list[0].clear()
                    #TODO

                #input_speech_system_event
                if events_list[1].isSet():
                    events_list[1].clear()
                    #TODO

                #input_state_charts_event
                if events_list[2].isSet():
                    events_list[2].clear()
                    #TODO

                # output_cameras_event
                if events_list[3].isSet():
                    events_list[3].clear()
                    #TODO

                #output_speech_system_event
                if events_list[4].isSet():
                    events_list[4].clear()
                    #TODO

                #output_state_charts_event
                if events_list[5].isSet():
                    events_list[5].clear()
                    #TODO
