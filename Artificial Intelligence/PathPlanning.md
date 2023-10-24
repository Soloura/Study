# [Path Planning](https://www.mathworks.com/discovery/path-planning.html)

Path planning lets an autonomous vehicle or a robot find the shortest and most obstacle-free path from a start to goal state. The path can be a set of states (position and/or orientation) or waypoints. Path planning requires a map of the environment along with start and goal states as input. The map can be represented in different ways such as grid maps, state spaces, and topological roadmap. Maps can be multilayered for adding bias to the path.

A few popular categories of path planning algorithms for autonomous vehicles include:

Grid-based search algorithms, which find a path based on minimum travel cost in a grid map. They can be used for applications such as mobile robots in a 2D environment. However, the memory requirements to implement grid-based algorithms could increase with the number of dimensions, such as for a 6-DOF robot manipulator.

Sampling-based search algorithms, which create a searchable tree by randomly sampling new nodes or robt configurations in a state space. Sampling-based algorithms can be suitable for high-dimensional search spaces such as those used to find a valid set of configurations for a robot arm to pick up an object. Generating dynamically feasible paths for various practical applications make sampling-based planning popular, even though it does not provide a complete solution.

Trajectory optimization algorithms, which formulate the path planning problem as an optimization problem that considers the desired vehicle performance, relevant constraints, and vehicle dynamics. Along with generating dynamically feasible trajectories, they can also be applied for online path planning in uncertain environments. However, depending on the complexity of the optimization problem, real-time planning can be prohibitive.

Path planning, along with perception (or vision) and control systems, comprise the three main building blocks of autonomous navigation for any robot or vehicle. Path planning adds autonomy in systems such as self-driving cars, robot manipulators, unmanned ground vehicles (UGVs), and unmanned aerial vehicles (UAVs).

# [Motion Planning](https://en.wikipedia.org/wiki/Motion_planning)

Motion planning, also path planning (also known as the navigation problem or the piano mover's problem) is a computational problem to find a sequence of valid configurations that moves the object from the source to destination. The term is used in computational geometry, computer animation, robotics and computer games.

---

### Reference
- What Is Path Planning, https://www.mathworks.com/discovery/path-planning.html, 2023-10-24-Tue.
- Motion Planning Wiki, https://en.wikipedia.org/wiki/Motion_planning, 2023-10-24-Tue.
