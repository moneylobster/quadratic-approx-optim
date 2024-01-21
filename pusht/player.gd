extends RigidBody2D

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
