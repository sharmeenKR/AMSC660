function monte_carlo_volume_simulation()
    close all;
    dims = [5, 10, 15, 20];  % Dimensions to evaluate
    N = 1e6;                 % Number of Monte Carlo samples (large for accuracy)

    % Initialize results table
    results = table('Size', [length(dims), 4], ...
                    'VariableTypes', {'double', 'double', 'double', 'double'}, ...
                    'VariableNames', {'Dimension', 'WayOneFraction', 'WayTwoFraction', 'ExactBallVol'});

    % Run simulations for each dimension
    for i = 1:length(dims)
        d = dims(i);

        % Exact volume of the unit ball
        exact_ball_vol = volume_Bd(d);

        % Monte Carlo: Way One (Cube Sampling)
        way_one_fraction = way_one_fraction_in_ball(d, N);

        % Monte Carlo: Way Two (Ball Sampling)
        way_two_fraction = way_two_fraction_in_cube_fixed(d, N);

        % Store results
        results.Dimension(i) = d;
        results.WayOneFraction(i) = way_one_fraction;
        results.WayTwoFraction(i) = way_two_fraction;
        results.ExactBallVol(i) = exact_ball_vol;
    end

    % Display results
    disp("Simulation Results:");
    disp(results);

    % Plot results
    plot_fraction_results(results);
end

function vol = volume_Bd(d)
    % Compute the exact volume of the unit ball in d dimensions
    vol = (2/d)*(pi^(d/2)) / gamma(d/2);
end

function fraction = way_one_fraction_in_ball(d, N)
    % Monte Carlo: Sampling uniformly in the cube
    X = rand(N, d) - 0.5;           % Uniform random points in the cube [-0.5, 0.5]^d
    r2 = sum(X.^2, 2);              % Squared distances from the origin
    count_inside_ball = sum(r2 <= 1); % Count points inside the ball
    fraction = count_inside_ball / N; % Fraction of cube inside ball
end

function fraction = way_two_fraction_in_cube_fixed(d, N)
    % Monte Carlo: Sampling uniformly in the ball
    Z = randn(N, d);                   % Generate random Gaussian points
    Z_norm = sqrt(sum(Z.^2, 2));       % Compute norms of Gaussian points
    Z_unit = Z ./ Z_norm;              % Normalize to unit sphere
    
    U = rand(N, 1);                    % Generate radii
    r = U.^(1 / d);                    % Scale radii using inverse transform
    X = Z_unit .* r;                   % Points within the unit ball

    % Check which points fall inside the cube [-0.5, 0.5]^d
    in_cube = all(abs(X) <= 0.5, 2);   % Logical array: 1 if inside cube, 0 otherwise
    count_inside_cube = sum(in_cube);  % Count points inside the cube

    % Compute Fraction (scaled by the volume of the ball)
    vol_ball = volume_Bd(d);           % Volume of the unit ball
    fraction = (count_inside_cube / N) * vol_ball; % Fraction of cube inside ball
end


function plot_fraction_results(results)
    figure;
    hold on;
    grid on;

    % Plot Way One (Cube Sampling)
    plot(results.Dimension, results.WayOneFraction, 'D--', 'Color', 'red', ...
        'LineWidth', 2, 'MarkerSize', 10, 'DisplayName', 'Way One (Cube Sampling)');
    
    % Plot Way Two (Ball Sampling)
    plot(results.Dimension, results.WayTwoFraction, '*:', 'Color', 'blue', ...
        'LineWidth', 2, 'MarkerSize', 12, 'DisplayName', 'Way Two (Ball Sampling)');
    
    xlabel('Dimension (d)');
    ylabel('Fraction');
    title('Fraction of Cube in Ball');
    legend show;
    hold off;
end
