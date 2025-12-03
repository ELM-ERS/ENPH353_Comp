rosservice call /gazebo/set_model_state "model_state:
  model_name: 'B1'
  pose:
    position: {x: -4.0, y: -2.28 , z: 0.5}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  twist:
    linear:  {x: 0.0, y: 0.0, z: 0.0}
    angular: {x: 0.0, y: 0.0, z: 0.0}
  reference_frame: 'world'"
