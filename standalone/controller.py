def initialize_system():
    print("System initialized.")

    court_dimensions = get_court_dimensions()

def fire():
    print("Firing the ball!")

    age = estimate_age()
    opponent_pos = get_opponent_position()

    if age < 20:
        print("easy")
        mode = "easy"
    elif age < 40:
        print("medium")
        mode = "medium"
    elif age < 60:
        print("hard")
        mode = "hard"
    else:
        print("impossible")
        mode = "impossible"

    target_pos = calculate_serve_course(mode, opponent_pos)
    launch_params = calculate_launch_parameters(target_pos)

    execute_serve(launch_params)

def return_ball():
    print("Returning the ball!")
    ball_trajectory = predict_ball_trajectory()
    intercept_point = calculate_intercept_point(ball_trajectory)

    move_params = plan_movement(intercept_point)
    move_robot(move_params)