extends Node2D

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
