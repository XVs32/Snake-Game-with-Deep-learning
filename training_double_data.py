from game import *

def generate_training_data(display, clock):
    training_data_x = []
    training_data_y = []
    training_games = 1000
    step_per_game = 2000
    
    counter_x = 0
    counter_y = 0
    
    for _ in tqdm(range(training_games)):
        snake_start_a, snake_position_a, score_a, snake_start_b, snake_position_b, score_b, apple_position= starting_positions()
    
        prev_apple_distance_a = apple_distance_from_snake(apple_position, snake_position_a)
        prev_apple_distance_b = apple_distance_from_snake(apple_position, snake_position_b)
        
        tem_training_data_x_a = []
        tem_training_data_y_a = []
        
        tem_training_data_x_b = []
        tem_training_data_y_b = []
        
        
        step_count = 0
        
        
        for _ in range(step_per_game):
            
            if collision_with_boundaries(snake_position_a[0]) == 1\
            or collision_with_boundaries(snake_position_b[0]) == 1\
            or collision_with_self(snake_position_a[0], snake_position_a[1:]) == 1\
            or collision_with_self(snake_position_a[0], snake_position_b) == 1\
            or collision_with_self(snake_position_b[0], snake_position_a) == 1\
            or collision_with_self(snake_position_b[0], snake_position_b[1:]) == 1:
                if score_a > score_b:
                    #print("score_a > score_b")
                    for i in tem_training_data_x_a:
                        training_data_x.append(i)
                        counter_x += 1
                    for i in tem_training_data_y_a:
                        training_data_y.append(i)
                        counter_y += 1
                else:
                    #print("score_b > score_a")
                    for i in tem_training_data_x_b:
                        training_data_x.append(i)
                        counter_x += 1
                    for i in tem_training_data_y_b:
                        training_data_y.append(i)
                        counter_y += 1
                        
                break
            
            angle_a, snake_direction_vector_a, apple_direction_vector_normalized_a, snake_direction_vector_normalized_a \
                = angle_with_apple(snake_position_a, apple_position)
            
            angle_b, snake_direction_vector_b, apple_direction_vector_normalized_b, snake_direction_vector_normalized_b \
                = angle_with_apple(snake_position_b, apple_position)
            
            #print("angle_a" + str(angle_a))
            #print("angle_b" + str(angle_b))
            
            #print("apple_direction_vector_normalized_a" + str(apple_direction_vector_normalized_a[0]) + " " + str(apple_direction_vector_normalized_a[1]))
            #print("apple_direction_vector_normalized_b" + str(apple_direction_vector_normalized_b[0]) + " " + str(apple_direction_vector_normalized_b[1]))
            
            
            direction_a, button_direction_a = generate_random_direction(snake_position_a, angle_a)
            direction_b, button_direction_b = generate_random_direction(snake_position_b, angle_b)
            
            
            #current_direction_vector, is_front_blocked, is_left_blocked, is_right_blocked = blocked_directions(snake_position)
            
            front_blocked_distance_a,\
            left_blocked_distance_a, right_blocked_distance_a\
                = blocked_directions(snake_position_a,snake_position_b)
            
            front_blocked_distance_b,\
            left_blocked_distance_b, right_blocked_distance_b\
                = blocked_directions(snake_position_b,snake_position_a)
            
            
            
            direction_a, button_direction_a, tem_training_data_y_a = generate_training_data_y(snake_position_a,
                                                                                          button_direction_a, direction_a,
                                                                                          tem_training_data_y_a, 
                                                                                          front_blocked_distance_a,
                                                                                          left_blocked_distance_a,
                                                                                          right_blocked_distance_a)
            
            direction_b, button_direction_b, tem_training_data_y_b = generate_training_data_y(snake_position_b, angle_b,
                                                                                          button_direction_b, direction_b,
                                                                                          tem_training_data_y_b, 
                                                                                          front_blocked_distance_b,
                                                                                          left_blocked_distance_b,
                                                                                          right_blocked_distance_b)
            #rule base snake to generate a training set
            #keep in mind, this is deep learning, not pure RL
            
            
            
            tem_training_data_x_a.append(
                [front_blocked_distance_a,\
                 left_blocked_distance_a, right_blocked_distance_a,\
                 snake_direction_vector_normalized_a[0], apple_direction_vector_normalized_a[0],\
                 snake_direction_vector_normalized_a[1], apple_direction_vector_normalized_a[1]])
            
            tem_training_data_x_b.append(
                [front_blocked_distance_b,\
                 left_blocked_distance_b, right_blocked_distance_b,\
                 snake_direction_vector_normalized_b[0], apple_direction_vector_normalized_b[0],\
                 snake_direction_vector_normalized_b[1], apple_direction_vector_normalized_b[1]])
            
            """snake_position, apple_position, score = play_game(snake_start, snake_position, apple_position,
                                                              button_direction, score, display, clock)"""
            
            
            
            #play_game
            #play_game_without_gui
            snake_position_a, score_a, snake_position_b, score_b, apple_position = play_game_without_gui(snake_start_a, snake_position_a,\
                                                                                             button_direction_a, score_a,\
                                                                                             snake_start_b, snake_position_b,\
                                                                                             button_direction_b, score_b,\
                                                                                             apple_position, display, clock)
            
    
    
    return training_data_x, training_data_y


def generate_training_data_y(snake_position, angle_with_apple, button_direction, direction, training_data_y,
                             is_front_blocked, is_left_blocked, is_right_blocked):
    if direction == -1:
        if is_left_blocked == 1:
            if is_front_blocked == 1 and is_right_blocked != 1:
                direction, button_direction = direction_vector(snake_position, 1)
                training_data_y.append([0, 0, 1])
            elif is_front_blocked != 1 and is_right_blocked == 1:
                direction, button_direction = direction_vector(snake_position, 0)
                training_data_y.append([0, 1, 0])
            elif is_front_blocked != 1 and is_right_blocked != 1:
                direction, button_direction = direction_vector(snake_position, 1)
                training_data_y.append([0, 0, 1])

        else:
            training_data_y.append([1, 0, 0])

    elif direction == 0:
        if is_front_blocked == 1:
            if is_left_blocked == 1 and is_right_blocked != 1:
                direction, button_direction = direction_vector(snake_position, 1)
                training_data_y.append([0, 0, 1])
            elif is_left_blocked != 1 and is_right_blocked == 1:
                direction, button_direction = direction_vector(snake_position, -1)
                training_data_y.append([1, 0, 0])
            elif is_left_blocked != 1 and is_right_blocked != 1:
                training_data_y.append([0, 0, 1])
                direction, button_direction = direction_vector(snake_position, 1)
        else:
            training_data_y.append([0, 1, 0])
    else:
        if is_right_blocked == 1:
            if is_left_blocked == 1 and is_front_blocked != 1:
                direction, button_direction = direction_vector(snake_position, 0)
                training_data_y.append([0, 1, 0])
            elif is_left_blocked != 1 and is_front_blocked == 1:
                direction, button_direction = direction_vector(snake_position, -1)
                training_data_y.append([1, 0, 0])
            elif is_left_blocked != 1 and is_front_blocked != 1:
                direction, button_direction = direction_vector(snake_position, -1)
                training_data_y.append([1, 0, 0])
        else:
            training_data_y.append([0, 0, 1])

    return direction, button_direction, training_data_y