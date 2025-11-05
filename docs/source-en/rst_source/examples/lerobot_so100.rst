Real-World RL on SO100 with LeRobot
===================================

This guide explains how to connect **RLinf** with the
`🤗 LeRobot <https://github.com/huggingface/lerobot>`_ stack in order to run
π\ :sub:`0.5`\ -based reinforcement learning directly on a
`SO-100 follower arm <https://github.com/TheRobotStudio/SO-ARM100>`_.

The integration adds a new environment backend (``simulator_type: lerobot``)
that wraps the LeRobot control stack and handles observation, action, and
reward preprocessing so that RLinf can drive the real robot with minimal glue
code.

Prerequisites
-------------

* A fully assembled and calibrated SO-100 follower arm (and optional wrist cameras)
* Python environment with RLinf and LeRobot (including the ``feetech`` extra) installed
* Access to a pretrained π\ :sub:`0`\ /π\ :sub:`0.5`\ model such as
  `RLinf/RLinf-Pi05-SFT <https://huggingface.co/RLinf/RLinf-Pi05-SFT>`_
* URDF describing the SO-100 mounting pose if you would like to use Cartesian
  control via inverse kinematics

Environment Configuration
-------------------------

Two ready-to-customise environment templates are provided:

* ``examples/embodiment/config/env/train/lerobot_so100.yaml``
* ``examples/embodiment/config/env/eval/lerobot_so100.yaml``

Update the following fields to match your setup:

``lerobot.robot.port``
    Serial port exposed by the Feetech USB adapter (e.g. ``/dev/ttyUSB0`` or ``/dev/ttyACM1``).

``lerobot.robot.id``
    Identifier that matches the calibration JSON file stored by LeRobot.

``lerobot.robot.cameras``
    Camera definitions (OpenCV indices or RealSense serial numbers).  The key names
    must match the cameras used when the dataset/model was collected.

``lerobot.primary_camera_key`` / ``lerobot.wrist_camera_keys``
    Observation keys produced by the LeRobot processor that RLinf should feed into the policy.

``lerobot.processor.inverse_kinematics``
    URDF path, workspace bounds, and step sizes used to convert π actions
    (delta end-effector poses) into SO-100 joint goals.  Set this block to
    ``null`` if you prefer to operate directly in joint space.

End-to-End Training Config
--------------------------

The file ``examples/embodiment/config/lerobot_so100_grpo_openpi05.yaml`` links
the new environment to the π\ :sub:`0.5`\ action model and configures GRPO
training with a single real-world environment (``group_size = num_group_envs = 1``).

Before launching training, edit the following placeholders:

* ``rollout.model_dir`` – path to the pretrained π model directory
* ``actor.checkpoint_load_path`` – same as above
* (Optionally) ``runner.logger.log_path`` – where rollouts, logs, and checkpoints should be saved

Running the Policy
------------------

Once the environment YAML has been customised, run RL training or evaluation
just like any other RLinf experiment.  For example, to launch GRPO fine-tuning:

.. code-block:: bash

   cd ${EMBODIED_PATH}/examples/embodiment
   python train.py \
       --config-name lerobot_so100_grpo_openpi05 \
       runner.max_epochs=50 \
       rollout.model_dir=/abs/path/to/RLinf-Pi05-SFT \
       actor.checkpoint_load_path=/abs/path/to/RLinf-Pi05-SFT

The rollout worker will stream camera images and proprioceptive states from
LeRobot into the π policy, convert the generated end-effector deltas to joint
targets via the LeRobot processor pipeline, and execute them on the arm.

Reward Shaping
--------------

Real-world reward functions vary from task to task.  You can plug in your own
reward classifier or handcrafted shaping by modifying the
``lerobot.processor.reward_classifier`` block or by extending the
``rlinf/envs/lerobot/so100_env.py`` environment.  When a reward classifier is
specified, RLinf automatically loads it through the LeRobot processor pipeline
and adds the predicted success reward to each transition.

Safety Notes
------------

* Always verify that the end-effector bounds in ``end_effector_bounds`` cover a
  safe workspace for your hardware setup.
* The example configuration leaves ``calibrate_on_start`` disabled so that the
  policy can be restarted without pushing a calibration routine.  Run the
  ``lerobot-setup-motors`` utility beforehand if calibration changes.
* Set ``lerobot.reset_joint_positions`` to a joint pose that keeps the arm clear
  of obstacles, or perform manual resets between episodes.
