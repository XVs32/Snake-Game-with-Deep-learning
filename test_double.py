import tensorflow.compat.v1 as tf
import tensorflow
from tensorflow.compat.v1.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from game import *
from training_data import generate_training_data

from keras.models import Sequential
from keras.layers import Dense

	
from game import *
from keras.models import model_from_json

def run_game_with_ML(model, display, clock):
	max_score = 3
	avg_score = 0
	
	training_data_x = []
	training_data_y = []
	test_games = 10
	steps_per_game = 2000

	counter_x = 0
	counter_y = 0

	for _ in range(test_games):
		snake_start_a, snake_position_a, score_a, snake_start_b, snake_position_b, score_b, apple_position= starting_positions()
		
		count_same_direction = 0
		
		training_data_x_a = []
		training_data_y_a = []
		
		training_data_x_b = []
		training_data_y_b = []

		for _ in range(steps_per_game):
			
			if collision_with_boundaries(snake_position_a[0]) == 1\
			or collision_with_boundaries(snake_position_b[0]) == 1\
			or collision_with_self(snake_position_a[0], snake_position_a[1:]) == 1\
			or collision_with_self(snake_position_a[0], snake_position_b) == 1\
			or collision_with_self(snake_position_b[0], snake_position_a) == 1\
			or collision_with_self(snake_position_b[0], snake_position_b[1:]) == 1:
				if score_a < 4 and score_b < 4:
					print("Info: didn't get any score")
					break
					
				if score_a > score_b:
					print("score_a > score_b")
					for i in training_data_x_a:
						training_data_x.append(i)
						counter_x += 1
					for i in training_data_y_a:
						training_data_y.append(i)
						counter_y += 1
				else:
					print("score_b > score_a")
					for i in training_data_x_b:
						training_data_x.append(i)
						counter_x += 1
					for i in training_data_y_b:
						training_data_y.append(i)
						counter_y += 1

				print("counter_x: " + str(counter_x))
				print("counter_y: " + str(counter_y))
				break
			
###########################################################################################################
################################################ snake A start ############################################
###########################################################################################################
			
			front_blocked_distance_a,\
			left_blocked_distance_a, right_blocked_distance_a,\
				= blocked_directions(snake_position_a,snake_position_b)
			
			
			angle_a, snake_direction_vector_a, apple_direction_vector_normalized_a, snake_direction_vector_normalized_a \
				= angle_with_apple(snake_position_a, apple_position)
			
			predictions = []
			
			predicted_direction = np.argmax(np.array(model.predict(np.array([front_blocked_distance_a, \
																			 left_blocked_distance_a, right_blocked_distance_a,\
																			 snake_direction_vector_normalized_a[0], apple_direction_vector_normalized_a[0],\
																			 snake_direction_vector_normalized_a[1], apple_direction_vector_normalized_a[1]]).reshape(-1,7)))) - 1
			
			new_direction = np.array(snake_position_a[0]) - np.array(snake_position_a[1])
			if predicted_direction == -1:
				new_direction = np.array([new_direction[1], -new_direction[0]])
				training_data_y_a.append([1, 0, 0])
			elif predicted_direction == 1:
				new_direction = np.array([-new_direction[1], new_direction[0]])
				training_data_y_a.append([0, 0, 1])
			elif predicted_direction == 0:
				training_data_y_a.append([0, 1, 0])
			else:
				print ("error: not defined predicted_direction")
			
			button_direction_a = generate_button_direction(new_direction)
			
			
			
			training_data_x_a.append(
				[front_blocked_distance_a,\
				 left_blocked_distance_a, right_blocked_distance_a,\
				 snake_direction_vector_normalized_a[0], apple_direction_vector_normalized_a[0],\
				 snake_direction_vector_normalized_a[1], apple_direction_vector_normalized_a[1]])
			
			
			
###########################################################################################################
################################################ snake A end ##############################################
###########################################################################################################
			#print("finish snake A")
###########################################################################################################
################################################ snake B start ############################################
###########################################################################################################
			
			front_blocked_distance_b,\
			left_blocked_distance_b, right_blocked_distance_b,\
				= blocked_directions(snake_position_b,snake_position_a)
			
			
			angle_b, snake_direction_vector_b, apple_direction_vector_normalized_b, snake_direction_vector_normalized_b \
				= angle_with_apple(snake_position_b, apple_position)
			
			#print("angle_a" + str(angle_a))
			#print("angle_b" + str(angle_b))

			#print("apple_direction_vector_normalized_a" + str(apple_direction_vector_normalized_a[0]) + " " + str(apple_direction_vector_normalized_a[1]))
			#print("apple_direction_vector_normalized_b" + str(apple_direction_vector_normalized_b[0]) + " " + str(apple_direction_vector_normalized_b[1]))

			
			predictions = []
			
			predicted_direction = np.argmax(np.array(model.predict(np.array([front_blocked_distance_b, \
																			 left_blocked_distance_b, right_blocked_distance_b,\
																			 snake_direction_vector_normalized_b[0], apple_direction_vector_normalized_b[0],\
																			 snake_direction_vector_normalized_b[1], apple_direction_vector_normalized_b[1]]).reshape(-1,7)))) - 1
			
			new_direction = np.array(snake_position_b[0]) - np.array(snake_position_b[1])
			if predicted_direction == -1:
				new_direction = np.array([new_direction[1], -new_direction[0]])
				training_data_y_b.append([1, 0, 0])
			elif predicted_direction == 1:
				new_direction = np.array([-new_direction[1], new_direction[0]])
				training_data_y_b.append([0, 0, 1])
			else:
				training_data_y_b.append([0, 1, 0])
			
			
			button_direction_b = generate_button_direction(new_direction)
			
			
			
			training_data_x_b.append(
				[front_blocked_distance_b,\
				 left_blocked_distance_b, right_blocked_distance_b,\
				 snake_direction_vector_normalized_b[0], apple_direction_vector_normalized_b[0],\
				 snake_direction_vector_normalized_b[1], apple_direction_vector_normalized_b[1]])
			
###########################################################################################################
################################################ snake B end ##############################################
###########################################################################################################
			#print("finish snake B")
			
			
			"""snake_position_a, score_a, snake_position_b, score_b, apple_position = play_game_without_gui(snake_start_a, snake_position_a, button_direction_a, score_a,\
																						snake_start_b, snake_position_b, button_direction_b, score_b,\
																						apple_position, display, clock)"""
			
			snake_position_a, score_a, snake_position_b, score_b, apple_position = play_game(snake_start_a, snake_position_a, button_direction_a, score_a,\
																						snake_start_b, snake_position_b, button_direction_b, score_b,\
																						apple_position, display, clock)
			
			
			
		
		avg_score += max_score
		
	return max_score, avg_score / 1000, training_data_x, training_data_y





display_width = 500
display_height = 500
pygame.init()
display=pygame.display.set_mode((display_width,display_height))
clock=pygame.time.Clock()
clock.tick(500000)
print("finish init")

training_data_x = []
training_data_y = []

for i in range(3):
	
	json_file = open('model_' + str(i) + '.json', 'r')
	loaded_json_model = json_file.read()
	model = model_from_json(loaded_json_model)
	model.load_weights('model_' + str(i) + '.h5')
	
	max_score, avg_score, training_data_x, training_data_y = run_game_with_ML(model,display,clock)
	print("Maximum score achieved is:  ", max_score)
	print("Average score achieved is:  ", avg_score)
	
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	model.fit((np.array(training_data_x).reshape(-1,7)),( np.array(training_data_y).reshape(-1,3)), batch_size = 10,epochs= 3)
	model.save_weights('model_' + str(i+1) + '.h5')
	model_json = model.to_json()
	with open('model_' + str(i+1) + '.json', 'w') as json_file:
		json_file.write(model_json)





