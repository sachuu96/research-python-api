from flask import Flask, request, render_template, jsonify
import numpy as np

app = Flask(__name__)

R = np.matrix(np.zeros([6, 6]))
Q = np.matrix(np.zeros([6, 6]))

# Get the Nearest values from the provided array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# This function returns all available actions in the state given as an argument
def available_actions(state):
    current_state_row = R[state,]        #getting the row in the R matrix which the state represent
    av_act = np.where(current_state_row >= 0) [1]   # getting the row indexs of the R matrix which are not -1
    return av_act

# This function chooses at random which action to be performed within the range
# of all the availalbe actions
def sample_next_actions(available_act):
    next_action = int(np.random.choice(available_act, 1)) # choosing the next action randomly from availalbe actions
    return next_action

# This function updates the Q matrix according to the path selected and the Q
# learning algorithm
def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1] # checking what is the index which holds the maximum amount

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma + max_value

# Run Q Learning algorithm to get the excersice schedule
def getexcericelist(fitnesevaluation):

    if (fitnesevaluation < 4):
        R = np.matrix([[-1, 0, 0, 0, 0, -1],
                       [0, -1, 0, 0, 0, -1],
                       [0, 0, -1, 0, 0, -1],
                       [0, 0, 0, -1, 0, -1],
                       [0, 0, 0, 0, -1, 100],
                       [0, 0, 0, 0, 0, -1]])
    elif (fitnesevaluation < 7):
        R = np.matrix([[-1, 0, 0, 0, 0, 100],
                       [0, -1, 0, 0, 0, 100],
                       [0, 0, -1, 0, 0, 100],
                       [0, 0, 0, -1, 0, -1],
                       [0, 0, 0, 0, -1, -1],
                       [0, 0, 0, 0, 0, -1]])
    else:
        R = np.matrix([[-1, 0, 0, 0, 0, 100],
                       [0, -1, 0, 0, 0, 100],
                       [0, 0, -1, 0, 0, 100],
                       [0, 0, 0, -1, 0, 100],
                       [0, 0, 0, 0, -1, 100],
                       [0, 0, 0, 0, 0, -1]])

    # Q matrix
    Q = np.matrix(np.zeros([6, 6]))

    # Gamma (learning parameter)
    gamma = 0.8

    # Initial state. (Usually to be chosen at random)
    initial_state = 1

    # Get available actions in the current state
    available_act = available_actions(initial_state)

    # Sample next action to be performed
    action = sample_next_actions(available_act)

    # Update Q matrix
    update(initial_state, action, gamma)

    # -----------------------------------------------------------------------------
    # Training
    for i in range(10000):
        current_state = np.random.randint(0, int(Q.shape[0]))
        available_act = available_actions(current_state)
        action = sample_next_actions(available_act)
        update(current_state, action, gamma)

    # Normalize the "trained" Q matrix
    print(Q / np.max(Q) * 100)

    # -------------------------------------------------------------------------------
    # Testing

    current_state = 0
    steps = [current_state]

    infinitloopstop = 1
    while current_state != 5:

        infinitloopstop = infinitloopstop + 1

        next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

        if next_step_index.shape[0] > 1:
            next_step_index = int(np.random.choice(next_step_index, size=1))
        else:
            next_step_index = int(next_step_index)

        steps.append(next_step_index)
        current_state = next_step_index

        if (infinitloopstop > 1000):
            break

    # print selected sequence of steps
    steps = list(dict.fromkeys(steps))
    return steps

# User fitness evaluation for running
def legsfitnessevaluation(user_age, user_gender, user_spenttime,age_array,womentime_array,mentime_array):

    timeindex = age_array.index(find_nearest(age_array, user_age))

    if (user_gender == 'male'):
        timearray = mentime_array
    else:
        timearray = womentime_array

    if (timearray[timeindex] >= user_spenttime):
        fitnesevaluation = 10
    elif (timearray[timeindex] * 2 <= user_spenttime):
        fitnesevaluation = 1
    else:
        fitnesevaluation = ((timearray[timeindex] - 2 * 2) / 10) - (
                user_spenttime - timearray[timeindex])
        fitnesevaluation = 10 - abs(fitnesevaluation)

    return fitnesevaluation

# User fitness evaluation for arms
def armsfitnessevaluation(user_age, user_gender, activitymeasurement,age_array,womentime_array,mentime_array):

    timeindex = age_array.index(find_nearest(age_array, user_age))

    if (user_gender == 'male'):
        timearray = mentime_array
    else:
        timearray = womentime_array

    if (timearray[timeindex] <= activitymeasurement):
        fitnesevaluation = 10
    else:
        fitnesevaluation = float(10) / timearray[timeindex]
        fitnesevaluation = fitnesevaluation * activitymeasurement

    return fitnesevaluation

# User fitness evaluation for abs
def absfitnessevaluation(user_age, activitymeasurement, age_array, hearratemin_array, hearratemax_array):

    timeindex = age_array.index(find_nearest(age_array, user_age))

    if(hearratemin_array[timeindex] > activitymeasurement):
        fitnesevaluation = 1
    elif(hearratemax_array[timeindex] < activitymeasurement):
        fitnesevaluation = 10
    else:
        fitnesevaluation = float(10) / (hearratemax_array[timeindex] - hearratemin_array[timeindex])
        fitnesevaluation = fitnesevaluation * (activitymeasurement - hearratemin_array[timeindex])

    return fitnesevaluation

# Get the excerise list based on the Q learning outcome
def excersielistforlegs(user_age, user_gender, activitymeasurement):

    age_array = [25, 35, 45, 55, 65]
    womentime_array = [13, 13.5, 14, 16, 17.5]
    mentime_array = [11, 11.5, 12, 13, 14]

    fitnesslevel = legsfitnessevaluation(user_age, user_gender, activitymeasurement,age_array,womentime_array,mentime_array)

    excersiselist = []
    excersiselist = getexcericelist(fitnesslevel)

    excersiselistbeginner = [
        {"exKey": 1, "name": "Side Hop","imageUrl":"https://cdn-ami-drupal.heartyhosting.com/sites/muscleandfitness.com/files/_main2_sidetosidehop.jpg","level":"beginner"},
        {"exKey": 2, "name": "Squats", "imageUrl":"https://media1.popsugar-assets.com/files/thumbor/_8L5Z28n57dzzdPT8pEV2Gmup50/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2016/03/04/675/n/1922398/2a4b0a04f46626f9_squat.jpg","level":"beginner"},
        {"exKey": 3, "name": "Backward Lunge", "imageUrl":"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/goblet-reverse-lunge-1441211036.jpg","level":"beginner"},
        {"exKey": 4, "name": "Wall Calf Raises", "imageUrl":"https://www.aliveandwellfitness.ca/wp-content/uploads/2015/02/Calf-Raise.jpg","level":"beginner"},
        {"exKey": 5, "name": "Sumo Squat Calf Raises with Wall", "imageUrl":"https://static.onecms.io/wp-content/uploads/sites/35/2012/06/16185952/plie-squat-calf-420_0.jpg","level":"beginner"},
        {"exKey": 6, "name": "Knee to Chest", "imageUrl":"https://www.verywellhealth.com/thmb/EpgYtdLIAW9nSX0pWozlcvNJSiY=/3525x2350/filters:no_upscale():max_bytes(150000):strip_icc()/Depositphotos_22103221_original-56a05fe85f9b58eba4b027a7.jpg","level":"beginner"}
    ]

    excersiselistintermediate = [
        {"exKey": 7,  "name": "Jumping Jacks","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/BLn2-1T1Yp-cgpoOU76QVkuhlpc/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2015/05/01/974/n/1922729/8a48e47672d474dc_c9d6640d1d97a449_jumping-jacks.xxxlarge/i/Jumping-Jacks.jpg","level":"medium"},
        {"exKey": 8,  "name": "Backward Lunge","imageUrl":"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/goblet-reverse-lunge-1441211036.jpg","level":"medium"},
        {"exKey": 9,  "name": "Wall Calf Raises","imageUrl":"https://www.aliveandwellfitness.ca/wp-content/uploads/2015/02/Calf-Raise.jpg","level":"medium"},
        {"exKey": 10, "name": "Calf Raise with Splayed Foot","imageUrl":"https://i.ytimg.com/vi/-M4-G8p8fmc/maxresdefault.jpg","level":"medium"},
        {"exKey": 11, "name": "Wall Sit","imageUrl":"http://s3.amazonaws.com/prod.skimble/assets/4655/skimble-workout-trainer-exercise-wall-sit-tip-toes-1_iphone.jpg","level":"medium"}
    ]

    excersiselistadvanced = [
        {"exKey": 12, "name": "Burpees","imageUrl":"https://www.cdn.spotebi.com/wp-content/uploads/2014/10/burpees-exercise-illustration.jpg","level":"hard"},
        {"exKey": 13, "name": "Squats","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/_8L5Z28n57dzzdPT8pEV2Gmup50/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2016/03/04/675/n/1922398/2a4b0a04f46626f9_squat.jpg","level":"hard"},
        {"exKey": 14, "name": "Curtsy Lunge","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/CAYVOgmZ__WZZpt1ReKTUOaSsY4/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2015/12/16/664/n/1922398/19ccad6a3187b053_Side-Lunge-Curtsy-Squat.jpg","level":"hard"},
        {"exKey": 15, "name": "Jumpying Squats","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/_gsXN6w15Fm3hLGdCX-rRUAv5vs/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2014/01/31/901/n/1922729/1545977b1743e558_Jump-Squat.jpg","level":"hard"},
        {"exKey": 16, "name": "Lying with Butterfly Stretch","imageUrl":"http://s3.amazonaws.com/prod.skimble/assets/1259100/image_iphone.jpg","level":"hard"},
        {"exKey": 17, "name": "Wall Sit","imageUrl":"http://s3.amazonaws.com/prod.skimble/assets/4655/skimble-workout-trainer-exercise-wall-sit-tip-toes-1_iphone.jpg","level":"hard"}
    ]

    excericesheudle = []

    for x in excersiselist:
        if(fitnesslevel < 4):
            excericesheudle.append(excersiselistbeginner[x])
        elif(fitnesslevel < 4):
            excericesheudle.append(excersiselistintermediate[x])
        else:
            excericesheudle.append(excersiselistadvanced[x])

    return excericesheudle

def excersielistforarms(user_age, user_gender, activitymeasurement):

    age_array = [25, 35, 45, 55, 65]
    womentime_array = [20, 19, 14, 10, 10]
    mentime_array = [28, 21, 16, 12, 11]

    fitnesslevel = armsfitnessevaluation(user_age, user_gender, activitymeasurement, age_array, womentime_array,
                                         mentime_array)

    excersiselist = []
    excersiselist = getexcericelist(fitnesslevel)

    excersiselistbeginner = [
        {"exKey": 18, "name": "Arm Raises","imageUrl":"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/0504-front-arm-raise-1441032989.jpg","level":"beginner"},
        {"exKey": 19, "name": "Side Arm Raise","imageUrl":"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/1205-armraise-1441032989.jpg","level":"beginner"},
        {"exKey": 20, "name": "Triceps Dips","imageUrl":"https://wiki-fitness.com/wp-content/uploads/2014/04/triceps-dips.jpg","level":"beginner"},
        {"exKey": 21, "name": "Arm Circles Clockwise","imageUrl":"http://www.backatsquarezero.com/wp-content/uploads/2017/10/Front-Raise-Circles.jpg","level":"beginner"},
        {"exKey": 22, "name": "Arm Circles CounterClockWise","imageUrl":"http://www.igophysio.co.uk/wp-content/uploads/2017/03/arm-circles.jpg","level":"beginner"},
        {"exKey": 23, "name": "Diamond Push-Ups","imageUrl":"https://cdn1.coachmag.co.uk/sites/coachmag/files/styles/insert_main_wide_image/public/2017/07/diamond-push-up-bradley-simmonds.jpg","level":"beginner"}
    ]

    excersiselistintermediate = [
        {"exKey": 24, "name": "Arm Raises","imageUrl":"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/0504-front-arm-raise-1441032989.jpg","level":"medium"},
        {"exKey": 25, "name": "Side Arm Raise","imageUrl":"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/1205-armraise-1441032989.jpg","level":"medium"},
        {"exKey": 26, "name": "Floor Tricep Dips","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/1Dp0qN0aVQd9lEApVaq93VAoG3k/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2014/04/25/773/n/1922729/71dfb70a7ef96be8_c58b14640c2032c6_triceps-dips.jpg.xxxlarge/i/Triceps-Dips.jpg","level":"medium"},
        {"exKey": 27, "name": "Military Push Ups","imageUrl":"https://bodylastics.com/wp-content/uploads/2018/08/resisted-decline-military-push-up.jpg","level":"medium"},
        {"exKey": 28, "name": "Alternative Hooks","imageUrl":"https://redefiningstrength.com/wp-content/uploads/2016/04/lunge-and-reach-e1461966612120.jpg","level":"medium"},
        {"exKey": 29, "name": "Push-up & Rotation","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/24DvTMytVexDVCjeWQi-IqjI8M8/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2014/09/19/009/n/1922729/76aca72c86654a36_11-Push-Up-Rotation/i/Push-Up-Rotation.jpg","level":"medium"}
    ]

    excersiselistadvanced = [
        {"exKey": 30, "name": "Arm Raises","imageUrl":"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/0504-front-arm-raise-1441032989.jpg","level":"hard"},
        {"exKey": 31, "name": "Side Arm Raise","imageUrl":"https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/1205-armraise-1441032989.jpg","level":"hard"},
        {"exKey": 32, "name": "Skipping WithOut Rope","imageUrl":"https://lifeinleggings.com/wp-content/uploads/2016/03/invisible-jump-rope-exercise.jpg","level":"hard"},
        {"exKey": 33, "name": "Burpees","imageUrl":"https://www.cdn.spotebi.com/wp-content/uploads/2014/10/burpees-exercise-illustration.jpg","level":"hard"},
        {"exKey": 34, "name": "Floor Tricep Dips","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/1Dp0qN0aVQd9lEApVaq93VAoG3k/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2014/04/25/773/n/1922729/71dfb70a7ef96be8_c58b14640c2032c6_triceps-dips.jpg.xxxlarge/i/Triceps-Dips.jpg","level":"hard"},
        {"exKey": 35, "name": "Military Push Ups","imageUrl":"https://bodylastics.com/wp-content/uploads/2018/08/resisted-decline-military-push-up.jpg","level":"hard"}
    ]

    excericesheudle = []

    for x in excersiselist:
        if (fitnesslevel < 4):
            excericesheudle.append(excersiselistbeginner[x])
        elif (fitnesslevel < 4):
            excericesheudle.append(excersiselistintermediate[x])
        else:
            excericesheudle.append(excersiselistadvanced[x])

    return excericesheudle

def excersielistforabs(user_age, activitymeasurement):

    age_array = [25, 35, 45, 55, 65]
    hearratemin_array = [98, 93, 88, 83, 78]
    hearratemax_array = [146, 138, 131, 123, 116]

    fitnesslevel = absfitnessevaluation(user_age, activitymeasurement, age_array, hearratemin_array,
                                        hearratemax_array)

    excersiselist = []
    excersiselist = getexcericelist(fitnesslevel)

    excersiselistbeginner = [
        {"exKey": 36, "name": "Abdominal Crunches","imageUrl":"https://3i133rqau023qjc1k3txdvr1-wpengine.netdna-ssl.com/wp-content/uploads/2014/07/Basic-Crunch_Exercise.jpg","level":"beginner"},
        {"exKey": 37, "name": "Russian Twist","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/Sy9BdUS395G5YSZL2ErQcNxPBSI/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2016/09/14/943/n/1922729/75a989e2_Core-Seated-Russian-Twist/i/Circuit-3-Exercise-4-Seated-Russian-Twist.jpg","level":"beginner"},
        {"exKey": 38, "name": "Mountain Climber","imageUrl":"https://rejuvage.com/wp-content/uploads/2019/07/iStock-957699448.jpg","level":"beginner"},
        {"exKey": 39, "name": "Heel Touch","imageUrl":"https://www.cdn.spotebi.com/wp-content/uploads/2014/10/alternate-heel-touchers-exercise-illustration.jpg","level":"beginner"},
        {"exKey": 40, "name": "Leg Raises","imageUrl":"https://www.cdn.spotebi.com/wp-content/uploads/2014/10/straight-leg-raise-exercise-illustration.jpg","level":"beginner"},
        {"exKey": 41, "name": "Cobra Stretch","imageUrl":"https://i.pinimg.com/736x/44/36/c4/4436c456e37af15167f4dd25b8967b0c.jpg","level":"beginner"}
    ]

    excersiselistintermediate = [
        {"exKey": 42, "name": "Crossover Crunch","imageUrl":"https://i.ytimg.com/vi/C4MbUFxLm2Y/hqdefault.jpg","level":"medium"},
        {"exKey": 43, "name": "Mountain Climber","imageUrl":"https://rejuvage.com/wp-content/uploads/2019/07/iStock-957699448.jpg","level":"medium"},
        {"exKey": 44, "name": "Bicycle Crunches","imageUrl":"https://bodylastics.com/wp-content/uploads/2018/08/Bicycle-Abs-Crunches.jpg","level":"medium"},
        {"exKey": 45, "name": "Heel Touch","imageUrl":"https://www.cdn.spotebi.com/wp-content/uploads/2014/10/alternate-heel-touchers-exercise-illustration.jpg","level":"medium"},
        {"exKey": 46, "name": "Leg Raises","imageUrl":"https://www.cdn.spotebi.com/wp-content/uploads/2014/10/straight-leg-raise-exercise-illustration.jpg","level":"medium"},
        {"exKey": 47, "name": "V-Up","imageUrl":"https://gethealthyu.com/wp-content/uploads/2014/09/V-Up_Exercise-2.jpg","level":"medium"}
    ]

    excersiselistadvanced = [
        {"exKey": 48, "name": "Push-Up & Rotation","imageUrl":"https://media1.popsugar-assets.com/files/thumbor/24DvTMytVexDVCjeWQi-IqjI8M8/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2014/09/19/009/n/1922729/76aca72c86654a36_11-Push-Up-Rotation/i/Push-Up-Rotation.jpg","level":"hard"},
        {"exKey": 49, "name": "Russian Twist","imageUrl":"https://i.pinimg.com/originals/70/c2/65/70c2652a92ba946bc6a37f13563fda03.jpg","level":"hard"},
        {"exKey": 50, "name": "Bicycle Crunches","imageUrl":"https://bodylastics.com/wp-content/uploads/2018/08/Bicycle-Abs-Crunches.jpg","level":"hard"},
        {"exKey": 51, "name": "Heel Touch","imageUrl":"https://www.cdn.spotebi.com/wp-content/uploads/2014/10/alternate-heel-touchers-exercise-illustration.jpg","level":"hard"},
        {"exKey": 52, "name": "Leg Raises","imageUrl":"https://www.cdn.spotebi.com/wp-content/uploads/2014/10/straight-leg-raise-exercise-illustration.jpg","level":"hard"},
        {"exKey": 53, "name": "V-Up","imageUrl":"https://gethealthyu.com/wp-content/uploads/2014/09/V-Up_Exercise-2.jpg","level":"hard"}
    ]

    excericesheudle = []

    for x in excersiselist:
        if (fitnesslevel < 4):
            excericesheudle.append(excersiselistbeginner[x])
        elif (fitnesslevel < 4):
            excericesheudle.append(excersiselistintermediate[x])
        else:
            excericesheudle.append(excersiselistadvanced[x])

    return excericesheudle

@app.route('/getlegsshedule', methods=['POST', 'GET'])
def generatesheduleforlegs():
    User_json = request.json

    user_age = User_json['userage']
    user_gender = User_json['usergender']
    activitymeasurement = User_json['activitymeasurement']

    weekshedulerlegs = {
        "Monday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Tuesday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Wednesday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Thursday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Friday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Saturday": excersielistforlegs(user_age, user_gender, activitymeasurement),
        "Sunday": excersielistforlegs(user_age, user_gender, activitymeasurement)
    }
    return jsonify(results=weekshedulerlegs)

@app.route('/getarmsshedule', methods=['POST', 'GET'])
def generatesheduleforarms():

    User_json = request.json

    user_age = User_json['userage']
    user_gender = User_json['usergender']
    activitymeasurement = User_json['activitymeasurement']

    weekshedulerlegs = {
        "Monday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Tuesday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Wednesday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Thursday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Friday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Saturday": excersielistforarms(user_age, user_gender, activitymeasurement),
        "Sunday": excersielistforarms(user_age, user_gender, activitymeasurement)
    }
    return jsonify(results=weekshedulerlegs)

@app.route('/getabsshedule', methods=['POST', 'GET'])
def generatesheduleforabs():

    User_json = request.json

    user_age = User_json['userage']
    activitymeasurement = User_json['activitymeasurement']

    weekshedulerlegs = {
        "Monday": excersielistforabs(user_age, activitymeasurement),
        "Tuesday": excersielistforabs(user_age, activitymeasurement),
        "Wednesday": excersielistforabs(user_age, activitymeasurement),
        "Thursday": excersielistforabs(user_age, activitymeasurement),
        "Friday": excersielistforabs(user_age, activitymeasurement),
        "Saturday": excersielistforabs(user_age, activitymeasurement),
        "Sunday": excersielistforabs(user_age, activitymeasurement)
    }

    return jsonify(results=weekshedulerlegs)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)