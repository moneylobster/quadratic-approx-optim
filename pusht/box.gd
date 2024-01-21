extends RigidBody2D


# Called when the node enters the scene tree for the first time.
func _ready():
	position=Vector2(randi_range(150,450),randi_range(150,450))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
