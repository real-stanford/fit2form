
grasp_object_template = """
    <robot name="block">
      <material name="blue">
        <color rgba="0.50 0.50 0.50 1.0"/>
      </material>

      <link name="block_base_link">
        <contact>
          <lateral_friction value="0.2"/>
          <rolling_friction value="0.001"/>
          <contact_cfm value="0.0"/>
          <contact_erp value="1.0"/>
        </contact>
        <inertial>
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <mass value="0.05"/>
          <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
        <visual>
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <geometry>
            <mesh filename="{0}" scale="1 1 1"/>
          </geometry>
          <material name="blue"/>
        </visual>
        <collision>
          <origin rpy="0 0 0" xyz="0 0 0"/>
          <geometry>
            <mesh filename="{1}" scale="1 1 1"/>
          </geometry>
        </collision>
      </link>
    </robot>
    """

# Original Source: https://github.com/nalt/wsg50-ros-pkg/tree/master/wsg_50_simulation
wsg50_template = """<?xml version="1.0"?>
<robot name="wsg_50">

  <!-- BASE LINK -->
  <link name="base_link">
    <inertial>
      <mass value="1.2" />
      <origin xyz="0 0 0" />
      <inertia  ixx="1.0" ixy="0.0"  ixz="0.0"  iyy="1.0"  iyz="0.0"  izz="1.0" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="assets/wsg50/WSG50_110.stl" scale="10 10 10"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="assets/wsg50/WSG50_110.stl" scale="10 10 10"/>
      </geometry>
    </collision>
  </link>
  
  <!--  GRIPPER GUIDE LEFT -->
  <link name="gripper_left">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="assets/wsg50/GUIDE_WSG50_110.stl" scale="0.01 0.01 0.01"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="base_joint_gripper_left" type="prismatic">
    <!-- upper limit prevent guide from touching the other guide. It also determines the breadth delta for finger-->
    <!-- Notice also that it is always somewhat open. And hence we don't see guides merging into each other-->
    <limit lower="0" upper="0.55" effort="100" velocity="1"/>
    <origin xyz="-0.55 0 0" rpy="0 0 0" />
    <parent link="base_link"/>
    <child link="gripper_left" />
    <axis xyz="1 0 0"/>
  </joint>

  <!--  GRIPPER MOUNT LEFT -->
  <link name="mount_left">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="assets/wsg50/WSG50_MOUNT_SIMPLE.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 0.9"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="assets/wsg50/WSG50_MOUNT_SIMPLE.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <joint name="guide_joint_mount_left" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="gripper_left"/>
    <child link="mount_left" />
  </joint>
 
  <!-- LEFT FINGER -->
  <link name="finger_left">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
    <contact>
        <lateral_friction value="1"/>
        <rolling_friction value="1"/>
        <spinning_friction value="1"/>
    </contact>

    <visual>
      <!-- 0 origin requires fingers to be slightly shifted now-->
      <origin xyz="0.0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="{left_finger_visual_mesh}" scale="{mesh_scale} {mesh_scale} {mesh_scale}"/>
      </geometry>
      <material name="finger_left_color">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="{left_finger_collision_mesh}" scale="{mesh_scale} {mesh_scale} {mesh_scale}"/>
      </geometry>
    </collision>
  </link>

  <joint name="mount_joint_finger_left" type="fixed">
    <origin xyz="{finger_x_offset} 0 {finger_z_offset}" rpy="0 0 0" />
    <parent link="mount_left"/>
    <child link="finger_left" />
  </joint>

  <!-- GRIPPER GUIDE RIGHT -->
  <link name="gripper_right">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>

    <visual>
      <origin xyz="0.0 0 0" rpy="3.14 0 0"/>
      <geometry>
        <mesh filename="assets/wsg50/GUIDE_WSG50_110.stl" scale="0.01 0.01 0.01"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="base_joint_gripper_right" type="prismatic">
    <limit lower="-0.55" upper="0" effort="100" velocity="1"/>
    <origin xyz="0.55 0 0" rpy="0 0 3.14159" />
    <parent link="base_link"/>
    <child link="gripper_right" />
    <axis xyz="-1 0 0"/>
  </joint>

  <!--  GRIPPER MOUNT RIGHT -->
  <link name="mount_right">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="assets/wsg50/WSG50_MOUNT_SIMPLE.obj" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 0.9"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="assets/wsg50/WSG50_MOUNT_SIMPLE.obj" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <joint name="guide_joint_mount_right" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="gripper_right"/>
    <child link="mount_right" />
  </joint>

  <!-- RIGHT FINGER -->
  <link name="finger_right">
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" />
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0" />
    </inertial>
    
    <contact>
        <lateral_friction value="1"/>
        <rolling_friction value="1"/>
        <spinning_friction value="1"/>
    </contact>

    <visual>
      <origin xyz="0.0 0 0" rpy="3.14 0 0" />
      <geometry>
        <mesh filename="{right_finger_visual_mesh}" scale="{mesh_scale} {mesh_scale} {mesh_scale}"/>
      </geometry>
      <material name="finger_right_color">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="3.14 0 0"/>
      <geometry>
        <mesh filename="{right_finger_collision_mesh}" scale="{mesh_scale} {mesh_scale} {mesh_scale}"/>
      </geometry>
    </collision>
  </link>

  <joint name="mount_joint_finger_right" type="fixed">
    <origin xyz="{finger_x_offset} 0 {finger_z_offset}" rpy="0 0 0" />
    <parent link="mount_right"/>
    <child link="finger_right" />
  </joint>
</robot>
    """
