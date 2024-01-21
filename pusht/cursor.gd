extends Polygon2D

var mouse=true

func _process(delta):
	# just snap to the mouse position instantly
	if mouse:
		position=get_global_mouse_position()
