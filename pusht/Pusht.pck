GDPC                �                                                                         X   res://.godot/exported/133200997/export-0b7195cb7ed30f0a5c89c23939dc183d-kineplayer.scn        �      :�Ȏ�n�X����z    P   res://.godot/exported/133200997/export-0da9c88c1f789cfa891cd44f0340838c-box.scn       �      ����c�Q�Ӝ��.    T   res://.godot/exported/133200997/export-2d98b62feff102ebd89597c79420336c-pushbox.scn �)      �      $<�gh�סTq��m�>    T   res://.godot/exported/133200997/export-36a25e342948d0ceacc500772b5412b3-player.scn  @       �      �W���ϕ��}�d��h2    T   res://.godot/exported/133200997/export-ac136afa712879512cd370a927c5ac61-cursor.scn  P      �      kG'E������񰸏�R    ,   res://.godot/global_script_class_cache.cfg  @1             ��Р�8���8~$}P�    D   res://.godot/imported/icon.svg-218a8f2b3041327d8a5756f3a245f83b.ctex�      �      �̛�*$q�*�́        res://.godot/uid_cache.bin   5      �       "~ ��H�䅂p��       res://box.gd              ��^_l�]���.�/M�       res://box.tscn.remap /      `       ��\�8(�I4���"       res://cursor.gd �      �       �3 �;�y�c�6�lg�       res://cursor.tscn.remap �/      c       �qL��ʭ�a��Q�       res://icon.svg  `1      �      C��=U���^Qu��U3       res://icon.svg.import   �      �       s�2�VC��ޝڶ��Ρ       res://kineplayer.gd �      t      ���n\��#5�=A�       res://kineplayer.tscn.remap �/      g       ��4Ȳ�uVkl�       res://player.gd        >      1^'ʺ��{�I��Nwm       res://player.tscn.remap `0      c       ������T�?�L���       res://project.binary�5      �      �=j����mNr��Nn       res://pushbox.gd@$      @      �$I?jo��*��U�a^       res://pushbox.tscn.remap�0      d       ���G3ǂ�����,�    extends RigidBody2D


# Called when the node enters the scene tree for the first time.
func _ready():
	position=Vector2(randi_range(150,450),randi_range(150,450))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
   RSRC                    PackedScene            ��������                                                  resource_local_to_scene    resource_name    custom_solver_bias    size    script 	   _bundled           local://RectangleShape2D_6ulra +         local://PackedScene_7qrry \         RectangleShape2D       
     �B  �B         PackedScene          	         names "         box 	   position    RigidBody2D 
   Polygon2D    color    polygon    CollisionShape2D    shape    	   variants       
         ��   ���>  �?  �?  �?%        H�  H�  H�  HB  HB  HB  HB  H�                node_count             nodes        ��������       ����                            ����                                 ����                   conn_count              conns               node_paths              editable_instances              version             RSRC extends Polygon2D

var mouse=true

func _process(delta):
	# just snap to the mouse position instantly
	if mouse:
		position=get_global_mouse_position()
        RSRC                    PackedScene            ��������                                                  resource_local_to_scene    resource_name 	   _bundled    script           local://PackedScene_1sag4 �          PackedScene          	         names "         cursor    color    polygon 
   Polygon2D    	   variants            �?          �?%            ��  ��          �?  �?          node_count             nodes        ��������       ����                          conn_count              conns               node_paths              editable_instances              version             RSRC      GST2   �   �      ����               � �        �  RIFF�  WEBPVP8L�  /������!"2�H�$�n윦���z�x����դ�<����q����F��Z��?&,
ScI_L �;����In#Y��0�p~��Z��m[��N����R,��#"� )���d��mG�������ڶ�$�ʹ���۶�=���mϬm۶mc�9��z��T��7�m+�}�����v��ح�m�m������$$P�����එ#���=�]��SnA�VhE��*JG�
&����^x��&�+���2ε�L2�@��		��S�2A�/E���d"?���Dh�+Z�@:�Gk�FbWd�\�C�Ӷg�g�k��Vo��<c{��4�;M�,5��ٜ2�Ζ�yO�S����qZ0��s���r?I��ѷE{�4�Ζ�i� xK�U��F�Z�y�SL�)���旵�V[�-�1Z�-�1���z�Q�>�tH�0��:[RGň6�=KVv�X�6�L;�N\���J���/0u���_��U��]���ǫ)�9��������!�&�?W�VfY�2���༏��2kSi����1!��z+�F�j=�R�O�{�
ۇ�P-�������\����y;�[ ���lm�F2K�ޱ|��S��d)é�r�BTZ)e�� ��֩A�2�����X�X'�e1߬���p��-�-f�E�ˊU	^�����T�ZT�m�*a|	׫�:V���G�r+�/�T��@U�N׼�h�+	*�*sN1e�,e���nbJL<����"g=O��AL�WO!��߈Q���,ɉ'���lzJ���Q����t��9�F���A��g�B-����G�f|��x��5�'+��O��y��������F��2�����R�q�):VtI���/ʎ�UfěĲr'�g�g����5�t�ۛ�F���S�j1p�)�JD̻�ZR���Pq�r/jt�/sO�C�u����i�y�K�(Q��7őA�2���R�ͥ+lgzJ~��,eA��.���k�eQ�,l'Ɨ�2�,eaS��S�ԟe)��x��ood�d)����h��ZZ��`z�պ��;�Cr�rpi&��՜�Pf��+���:w��b�DUeZ��ڡ��iA>IN>���܋�b�O<�A���)�R�4��8+��k�Jpey��.���7ryc�!��M�a���v_��/�����'��t5`=��~	`�����p\�u����*>:|ٻ@�G�����wƝ�����K5�NZal������LH�]I'�^���+@q(�q2q+�g�}�o�����S߈:�R�݉C������?�1�.��
�ڈL�Fb%ħA ����Q���2�͍J]_�� A��Fb�����ݏ�4o��'2��F�  ڹ���W�L |����YK5�-�E�n�K�|�ɭvD=��p!V3gS��`�p|r�l	F�4�1{�V'&����|pj� ߫'ş�pdT�7`&�
�1g�����@D�˅ �x?)~83+	p �3W�w��j"�� '�J��CM�+ �Ĝ��"���4� ����nΟ	�0C���q'�&5.��z@�S1l5Z��]�~L�L"�"�VS��8w.����H�B|���K(�}
r%Vk$f�����8�ڹ���R�dϝx/@�_�k'�8���E���r��D���K�z3�^���Vw��ZEl%~�Vc���R� �Xk[�3��B��Ğ�Y��A`_��fa��D{������ @ ��dg�������Mƚ�R�`���s����>x=�����	`��s���H���/ū�R�U�g�r���/����n�;�SSup`�S��6��u���⟦;Z�AN3�|�oh�9f�Pg�����^��g�t����x��)Oq�Q�My55jF����t9����,�z�Z�����2��#�)���"�u���}'�*�>�����ǯ[����82һ�n���0�<v�ݑa}.+n��'����W:4TY�����P�ר���Cȫۿ�Ϗ��?����Ӣ�K�|y�@suyo�<�����{��x}~�����~�AN]�q�9ޝ�GG�����[�L}~�`�f%4�R!1�no���������v!�G����Qw��m���"F!9�vٿü�|j�����*��{Ew[Á��������u.+�<���awͮ�ӓ�Q �:�Vd�5*��p�ioaE��,�LjP��	a�/�˰!{g:���3`=`]�2��y`�"��N�N�p���� ��3�Z��䏔��9"�ʞ l�zP�G�ߙj��V�>���n�/��׷�G��[���\��T��Ͷh���ag?1��O��6{s{����!�1�Y�����91Qry��=����y=�ٮh;�����[�tDV5�chȃ��v�G ��T/'XX���~Q�7��+[�e��Ti@j��)��9��J�hJV�#�jk�A�1�^6���=<ԧg�B�*o�߯.��/�>W[M���I�o?V���s��|yu�xt��]�].��Yyx�w���`��C���pH��tu�w�J��#Ef�Y݆v�f5�e��8��=�٢�e��W��M9J�u�}]釧7k���:�o�����Ç����ս�r3W���7k���e�������ϛk��Ϳ�_��lu�۹�g�w��~�ߗ�/��ݩ�-�->�I�͒���A�	���ߥζ,�}�3�UbY?�Ӓ�7q�Db����>~8�]
� ^n׹�[�o���Z-�ǫ�N;U���E4=eȢ�vk��Z�Y�j���k�j1�/eȢK��J�9|�,UX65]W����lQ-�"`�C�.~8ek�{Xy���d��<��Gf�ō�E�Ӗ�T� �g��Y�*��.͊e��"�]�d������h��ڠ����c�qV�ǷN��6�z���kD�6�L;�N\���Y�����
�O�ʨ1*]a�SN�=	fH�JN�9%'�S<C:��:`�s��~��jKEU�#i����$�K�TQD���G0H�=�� �d�-Q�H�4�5��L�r?����}��B+��,Q�yO�H�jD�4d�����0*�]�	~�ӎ�.�"����%
��d$"5zxA:�U��H���H%jس{���kW��)�	8J��v�}�rK�F�@�t)FXu����G'.X�8�KH;���[             [remap]

importer="texture"
type="CompressedTexture2D"
uid="uid://dql305nd6me47"
path="res://.godot/imported/icon.svg-218a8f2b3041327d8a5756f3a245f83b.ctex"
metadata={
"vram_texture": false
}
                extends CharacterBody2D

@export var K=1
var mouse=true
var target=Vector2.ZERO

func _ready():
	if mouse:
		position=get_global_mouse_position()
	else:
		position=Vector2.ZERO

func _physics_process(delta):
	if mouse:
		target=get_local_mouse_position()
	velocity=K*target
	move_and_slide()

func _process(delta):
	if Input.is_action_pressed("quit"):
		get_tree().quit()
            RSRC                    PackedScene            ��������                                                  resource_local_to_scene    resource_name    custom_solver_bias    radius    script 	   _bundled       Script    res://kineplayer.gd ��������      local://CircleShape2D_ffvvi U         local://PackedScene_2w0x6 s         CircleShape2D             PackedScene          	         names "         kineplayer 	   position    collision_mask    motion_mode    platform_on_leave    platform_floor_layers    script    CharacterBody2D    CollisionShape2D    shape    debug_color 
   Polygon2D    polygon    	   variants    	   
         ��                   (    ���                               ��?��3?���>%             �  ��  �   �  ��   �   �   �   @   �  �@  ��  A  ��   A   @   A  �@  A   A  �@  A  �@   A  �?   A  @�   A  ��  �@   �   @   �      node_count             nodes     '   ��������       ����                                                          ����   	      
                        ����                   conn_count              conns               node_paths              editable_instances              version             RSRC           extends RigidBody2D

@export var initpos=Vector2()
@export var K=1
@export var maxval=1000
@export var Kp=10
@export var Kd=0.1

var F_t1 = Vector2.ZERO


# Called when the node enters the scene tree for the first time.
func _ready():
	#position=get_global_mouse_position()
	pass

func _process(delta):
	pass

func _physics_process(delta):
	#position=get_global_mouse_position()
	#print(get_local_mouse_position())
	pass

func _integrate_forces(state):
	var dt = state.step
	var vel = state.linear_velocity
	var vel_clipped = Vector2(sign(vel[0])*min(abs(vel[0]),maxval), sign(vel[1])*min(abs(vel[1]),maxval))
	var a = (get_local_mouse_position()-vel*dt)/pow(dt,2)
	var F = a/K
	var F_clipped = Vector2(sign(F[0])*min(abs(F[0]),maxval), sign(F[1])*min(abs(F[1]),maxval))
	var deriv = (F-F_t1)
	# move toward mouse
	state.apply_central_force(F)
	#if state.get_contact_count()!=0:
		#var fcont=state.get_contact_impulse(0)
		#print(fcont)
		#state.apply_central_force(-fcont)
	F_t1=F
	#print(Kp*get_local_mouse_position())
	#state.apply_central_force(Kp*get_local_mouse_position())
	pass
  RSRC                    PackedScene            ��������                                                  resource_local_to_scene    resource_name    custom_solver_bias    radius    script 	   _bundled           local://CircleShape2D_ffvvi *         local://PackedScene_v5hdr H         CircleShape2D             PackedScene          	         names "         player 	   position    RigidBody2D    CollisionShape2D    shape    debug_color 
   Polygon2D    polygon    	   variants       
         ��                 ��?��3?���>%             �  ��  �   �  ��   �   �   �   @   �  �@  ��  A  ��   A   @   A  �@  A   A  �@  A  �@   A  �?   A  @�   A  ��  �@   �   @   �      node_count             nodes        ��������       ����                            ����                                 ����                   conn_count              conns               node_paths              editable_instances              version             RSRC    extends Node2D

var tcp=StreamPeerTCP.new()
var auto=false

# Called when the node enters the scene tree for the first time.
func _ready():
	# parse cmdline args
	var args=OS.get_cmdline_user_args()
	print(args)
	if args.has("--user"):
		print("User mode")
		pass # user controlled via mouse.
	elif args.has("--auto"):
		print("Auto mode")
		auto=true
		$cursor.mouse=false
		$kineplayer.mouse=false
	# connect to server
	tcp.connect_to_host("127.0.0.1",65432)
	tcp.set_no_delay(true)
	print(tcp.poll())

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _physics_process(delta):
	# send current state
	var mousepos=get_global_mouse_position()
	var playerpos=$kineplayer.position
	var boxpos=$box.position
	var boxtheta=$box.rotation_degrees
	
	var message=[int(mousepos[0]),
				int(mousepos[1]),
				int(playerpos[0]),
				int(playerpos[1]),
				int(boxpos[0]),
				int(boxpos[1]),
				int(boxtheta)]
	# send a small 1-byte thing before the actual message
	tcp.put_data("start".to_ascii_buffer())
	# send message
	for i in message:
		tcp.put_64(i)
	if auto:
		# if available, get command
		if tcp.get_available_bytes():
			var cmdx=tcp.get_64()
			var cmdy=tcp.get_64()
			print(cmdx, cmdy)
			var targetpos=Vector2(cmdx, cmdy)
			$cursor.position=targetpos
			$kineplayer.target=targetpos-$kineplayer.position
RSRC                    PackedScene            ��������                                                  resource_local_to_scene    resource_name 	   _bundled    script       Script    res://pushbox.gd ��������   PackedScene    res://box.tscn ��d�6$#   Script    res://box.gd ��������   PackedScene    res://kineplayer.tscn �I��/�"   PackedScene    res://cursor.tscn �M��%�r+   Script    res://cursor.gd ��������      local://PackedScene_yrab7 �         PackedScene          	         names "         pushbox    script    Node2D    box 	   position    mass    continuous_cd    linear_damp    angular_damp    kineplayer    K    cursor    polygon    goal    color 
   Polygon2D    	   variants                          
     �C  ,C      A                        
              
            %            @�  @�          @@  @@             
     �C  �C         �?    ��?%        �  ��  �  �A  B  �A  B  ��      node_count             nodes     ?   ��������       ����                      ���                                                         ���	               
                  ���   	         
                           ����                               conn_count              conns               node_paths              editable_instances              version             RSRC           [remap]

path="res://.godot/exported/133200997/export-0da9c88c1f789cfa891cd44f0340838c-box.scn"
[remap]

path="res://.godot/exported/133200997/export-ac136afa712879512cd370a927c5ac61-cursor.scn"
             [remap]

path="res://.godot/exported/133200997/export-0b7195cb7ed30f0a5c89c23939dc183d-kineplayer.scn"
         [remap]

path="res://.godot/exported/133200997/export-36a25e342948d0ceacc500772b5412b3-player.scn"
             [remap]

path="res://.godot/exported/133200997/export-2d98b62feff102ebd89597c79420336c-pushbox.scn"
            list=Array[Dictionary]([])
     <svg height="128" width="128" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="2" width="124" height="124" rx="14" fill="#363d52" stroke="#212532" stroke-width="4"/><g transform="scale(.101) translate(122 122)"><g fill="#fff"><path d="M105 673v33q407 354 814 0v-33z"/><path fill="#478cbf" d="m105 673 152 14q12 1 15 14l4 67 132 10 8-61q2-11 15-15h162q13 4 15 15l8 61 132-10 4-67q3-13 15-14l152-14V427q30-39 56-81-35-59-83-108-43 20-82 47-40-37-88-64 7-51 8-102-59-28-123-42-26 43-46 89-49-7-98 0-20-46-46-89-64 14-123 42 1 51 8 102-48 27-88 64-39-27-82-47-48 49-83 108 26 42 56 81zm0 33v39c0 276 813 276 813 0v-39l-134 12-5 69q-2 10-14 13l-162 11q-12 0-16-11l-10-65H447l-10 65q-4 11-16 11l-162-11q-12-3-14-13l-5-69z"/><path d="M483 600c3 34 55 34 58 0v-86c-3-34-55-34-58 0z"/><circle cx="725" cy="526" r="90"/><circle cx="299" cy="526" r="90"/></g><g fill="#414042"><circle cx="307" cy="532" r="60"/><circle cx="717" cy="532" r="60"/></g></g></svg>
             ��d�6$#   res://box.tscn�M��%�r+   res://cursor.tscn��/#8vGs   res://icon.svg�I��/�"   res://kineplayer.tscn�ȡ����c   res://player.tscnh�@*S�   res://pushbox.tscn               ECFG
      application/config/name         Pusht      application/run/main_scene         res://pushbox.tscn     application/config/features(   "         4.2    GL Compatibility       application/config/icon         res://icon.svg  "   display/window/size/viewport_width         #   display/window/size/viewport_height         
   input/quit�              deadzone      ?      events              InputEventKey         resource_local_to_scene           resource_name             device         	   window_id             alt_pressed           shift_pressed             ctrl_pressed          meta_pressed          pressed           keycode           physical_keycode    @ 	   key_label             unicode           echo          script      !   physics/2d/default_gravity_vector              #   rendering/renderer/rendering_method         gl_compatibility*   rendering/renderer/rendering_method.mobile         gl_compatibility            