import pygame
import random
import time
import math
from tqdm import tqdm
import numpy as np


def display_snake(snake_position, display, snake_id):
    for position in snake_position:
        if snake_id == 1:
            pygame.draw.rect(display, (255, 0, 0), pygame.Rect(position[0], position[1], 10, 10))
        else:
            pygame.draw.rect(display, (0, 0, 255), pygame.Rect(position[0], position[1], 10, 10))

def display_apple(apple_position, display):
    pygame.draw.rect(display, (0, 255, 0), pygame.Rect(apple_position[0], apple_position[1], 10, 10))

def starting_positions():
    
    snake_start_a = [50, 100]
    snake_position_a = [[50, 100], [40, 100], [30, 100]]
    
    snake_start_b = [450, 400]
    snake_position_b = [[450, 400], [460, 400], [470, 400]]
    
    apple_position = [random.randrange(5, 45) * 10, random.randrange(15, 35) * 10]
    score_a = 3
    score_b = 3
    
    return snake_start_a, snake_position_a, score_a, snake_start_b, snake_position_b, score_b, apple_position


def apple_distance_from_snake(apple_position, snake_position):
    return np.linalg.norm(np.array(apple_position) - np.array(snake_position[0]))


def generate_snake(snake_start, snake_position, apple_position, button_direction, score):
    if button_direction == 1:
        snake_start[0] += 10
    elif button_direction == 0:
        snake_start[0] -= 10
    elif button_direction == 2:
        snake_start[1] += 10
    else:
        snake_start[1] -= 10

    if snake_start == apple_position:
        apple_position, score = collision_with_apple(apple_position, score)
        snake_position.insert(0, list(snake_start))

    else:
        snake_position.insert(0, list(snake_start))
        snake_position.pop()

    return snake_position, apple_position, score


def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    score += 1
    return apple_position, score


def collision_with_boundaries(snake_start):
    if snake_start[0] >= 500 or snake_start[0] < 0 or snake_start[1] >= 500 or snake_start[1] < 0:
        return 1
    else:
        return 0

def collision_with_self(snake_start, snake_position):
    # snake_start = snake_position[0]
    if snake_start in snake_position[0:]:
        return 1
    else:
        return 0


def blocked_directions(snake_position, rival_snake_position):
    
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    
    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])
    
    
    front_blocked_distance          = get_direction_distance(snake_position, rival_snake_position, current_direction_vector)
    #front_left_blocked_distance     = get_direction_distance(snake_position, rival_snake_position, current_direction_vector+left_direction_vector)
    #front_right_blocked_distance    = get_direction_distance(snake_position, rival_snake_position, current_direction_vector+right_direction_vector)
    
    left_blocked_distance           = get_direction_distance(snake_position, rival_snake_position, left_direction_vector)
    right_blocked_distance          = get_direction_distance(snake_position, rival_snake_position, right_direction_vector)
    
    #rear_left_blocked_distance      = get_direction_distance(snake_position, rival_snake_position, -current_direction_vector+left_direction_vector)
    #rear_right_blocked_distance     = get_direction_distance(snake_position, rival_snake_position, -current_direction_vector+right_direction_vector)
    
    return front_blocked_distance, \
           left_blocked_distance, right_blocked_distance



def get_direction_distance(snake_position, rival_snake_position, current_direction_vector):
    
    i=0
    next_step = snake_position[0]
    while True:
        i+=1
        next_step += current_direction_vector
        
        if collision_with_boundaries(next_step) == 1 \
        or collision_with_self(next_step.tolist(), snake_position) == 1 \
        or collision_with_self(next_step.tolist(), rival_snake_position) == 1:
            return (51 - i)/50

def generate_random_direction(snake_position, angle_with_apple):
    direction = 0
    if angle_with_apple > 89:
        direction = 1
    elif angle_with_apple < -89:
        direction = -1
    else:
        direction = 0

    return direction_vector(snake_position, direction)


def direction_vector(snake_position, direction):
    
    current_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])
    new_direction = current_direction_vector

    if direction == -1:
        new_direction = np.array([current_direction_vector[1], -current_direction_vector[0]]) #left_direction_vector
    if direction == 1:
        new_direction = np.array([-current_direction_vector[1], current_direction_vector[0]]) #right_direction_vector

    button_direction = generate_button_direction(new_direction)

    return direction, button_direction


def generate_button_direction(new_direction):
    button_direction = 0
    if new_direction.tolist() == [10, 0]:
        button_direction = 1
    elif new_direction.tolist() == [-10, 0]:
        button_direction = 0
    elif new_direction.tolist() == [0, 10]:
        button_direction = 2
    else:
        button_direction = 3

    return button_direction


def angle_with_apple(snake_position, apple_position):
    apple_direction_vector = np.array(apple_position) - np.array(snake_position[0])
    snake_direction_vector = np.array(snake_position[0]) - np.array(snake_position[1])

    norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
    norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 10
    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 10

    apple_direction_vector_normalized = apple_direction_vector / norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector / norm_of_snake_direction_vector
    angle = math.atan2(
        apple_direction_vector_normalized[1] * snake_direction_vector_normalized[0] - apple_direction_vector_normalized[
            0] * snake_direction_vector_normalized[1],
        apple_direction_vector_normalized[1] * snake_direction_vector_normalized[1] + apple_direction_vector_normalized[
            0] * snake_direction_vector_normalized[0]) / math.pi
    return angle * 180, snake_direction_vector, apple_direction_vector_normalized, snake_direction_vector_normalized


def play_game(snake_start_a, snake_position_a, button_direction_a, score_a,\
              snake_start_b, snake_position_b, button_direction_b, score_b,\
              apple_position, display, clock):
    
    #print("into play game")
    
    crashed = False
    while crashed is not True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        display.fill((255, 255, 255))

        display_apple(apple_position, display)
        display_snake(snake_position_a, display,1)
        display_snake(snake_position_b, display,2)

        snake_position_a, apple_position, score_a = generate_snake(snake_start_a, snake_position_a, apple_position,
                                                               button_direction_a, score_a)
        
        snake_position_b, apple_position, score_b = generate_snake(snake_start_b, snake_position_b, apple_position,
                                                               button_direction_b, score_b)
        
        pygame.display.set_caption("SCORE_a: " + str(score_a) +" SCORE_b: " + str(score_b))
        pygame.display.update()
        clock.tick(10)
        
        
        return snake_position_a, score_a, snake_position_b, score_b, apple_position
    
def play_game_without_gui(snake_start_a, snake_position_a, button_direction_a, score_a,\
              snake_start_b, snake_position_b, button_direction_b, score_b,\
              apple_position, display, clock):
    
    #print("into play game")
    
    crashed = False
    while crashed is not True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                crashed = True
        #display.fill((255, 255, 255))

        #display_apple(apple_position, display)
        #display_snake(snake_position_a, display,1)
        #display_snake(snake_position_b, display,2)

        snake_position_a, apple_position, score_a = generate_snake(snake_start_a, snake_position_a, apple_position,
                                                               button_direction_a, score_a)
        
        snake_position_b, apple_position, score_b = generate_snake(snake_start_b, snake_position_b, apple_position,
                                                               button_direction_b, score_b)
        
        #pygame.display.set_caption("SCORE_a: " + str(score_a) +" SCORE_b: " + str(score_b))
        #pygame.display.update()
        clock.tick(500000)
        
        
        return snake_position_a, score_a, snake_position_b, score_b, apple_position