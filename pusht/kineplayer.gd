extends CharacterBody2D

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
