"""The drive-mode input mapping (site/assets/drive.js) that the live pages share. Run with:  python -m unittest discover tests"""
import json
import os
import shutil
import subprocess
import unittest

NODE = shutil.which("node")
HERE = os.path.dirname(os.path.abspath(__file__))


def run_js(expression):
    script = "const D=require('../../site/assets/drive.js');console.log(JSON.stringify(%s));" % expression
    return json.loads(subprocess.run([NODE, "-e", script], cwd=HERE, capture_output=True, text=True, check=True).stdout)


@unittest.skipIf(NODE is None, "node is not installed")
class TestDriveMapping(unittest.TestCase):
    def test_keys_give_full_stick(self):
        self.assertEqual(run_js("D.fractions(new Set(['fwd','turnl']),null)"), [1, 0, 1])
        back_right = run_js("D.fractions(new Set(['back','right','turnr']),null)")  # back and right together are on the unit circle
        self.assertAlmostEqual(back_right[0], -(0.5 ** 0.5))
        self.assertAlmostEqual(back_right[1], -(0.5 ** 0.5))
        self.assertEqual(back_right[2], -1)

    def test_opposite_keys_cancel(self):
        self.assertEqual(run_js("D.fractions(new Set(['fwd','back','left','right']),null)"), [0, 0, 0])

    def test_diagonal_is_no_faster_than_straight(self):
        f = run_js("D.fractions(new Set(['fwd','left']),null)")
        self.assertAlmostEqual(f[0] ** 2 + f[1] ** 2, 1.0)
        self.assertAlmostEqual(f[0], f[1])

    def test_stick_dead_zone_and_rescaling(self):
        dead = run_js("D.DEAD")
        self.assertEqual(run_js("D.stick(%f)" % (dead * 0.99)), 0)
        self.assertAlmostEqual(run_js("D.stick(1)"), 1.0)
        self.assertAlmostEqual(run_js("D.stick(-1)"), -1.0)
        self.assertAlmostEqual(run_js("D.stick(%f)" % ((1 + dead) / 2)), 0.5)  # halfway between the dead zone and full

    def test_pad_axes_signs(self):
        # standard layout: stick up is y = -1 (forward), stick left is x = -1 (left, and turn left on the right stick)
        self.assertEqual(run_js("D.fractions(new Set(),[0,-1,0,0])"), [1, 0, 0])
        self.assertEqual(run_js("D.fractions(new Set(),[-1,0,0,0])"), [0, 1, 0])
        self.assertEqual(run_js("D.fractions(new Set(),[0,0,-1,0])"), [0, 0, 1])

    def test_a_moved_stick_beats_keys_but_a_resting_one_does_not(self):
        self.assertEqual(run_js("D.fractions(new Set(['fwd']),[0.05,0.05,0,0])"), [1, 0, 0])
        self.assertEqual(run_js("D.fractions(new Set(['fwd']),[0,1,0,0])"), [-1, 0, 0])

    def test_pose_signs(self):
        # flight-stick sense: forward pitches the nose down (+), right rolls the right side down (+), Q yaws left (+)
        self.assertEqual(run_js("D.poseFractions(new Set(['fwd','right','turnl','up']),null)"), [1, 1, 1, 1])
        self.assertEqual(run_js("D.poseFractions(new Set(['back','left','turnr','downh']),null)"), [-1, -1, -1, -1])
        self.assertEqual(run_js("D.poseFractions(new Set(),[1,-1,-1,-1])"), [1, 1, 1, 1])      # stick right, up; right stick left and up (up raises the body)
        self.assertEqual(run_js("D.poseFractions(new Set(['fwd']),[0.05,0.05,0.05,0.05])"), [0, 1, 0, 0])  # resting sticks do not override keys

    def test_pose_scaling_and_release(self):
        self.assertEqual(run_js("D.scalePose([1,-1,0.5,-1],{body_roll:0.15,body_pitch:0.15,body_yaw:0.2,body_height:0.04})"), [0.15, -0.15, 0.1, -0.04])
        self.assertEqual(run_js("D.poseFractions(new Set(),null)"), [0, 0, 0, 0])

    def test_height_fraction_signs(self):
        self.assertEqual(run_js("D.heightFraction(new Set(['up']),null)"), 1)
        self.assertEqual(run_js("D.heightFraction(new Set(['downh']),null)"), -1)
        self.assertEqual(run_js("D.heightFraction(new Set(['up','downh']),null)"), 0)
        self.assertEqual(run_js("D.heightFraction(new Set(),[0,0,0,-1])"), 1)    # right stick up is negative y on a standard pad
        self.assertEqual(run_js("D.heightFraction(new Set(),[0,0,0,1])"), -1)
        self.assertEqual(run_js("D.heightFraction(new Set(['up']),[0,0,0,0.05])"), 1)   # a resting stick does not override keys

    def test_step_height_moves_at_the_rate_and_stays_in_range(self):
        self.assertAlmostEqual(run_js("D.stepHeight(0.16,1,0.04,0.5,0.1,0.32)"), 0.18)
        self.assertAlmostEqual(run_js("D.stepHeight(0.16,-0.5,0.04,1,0.1,0.32)"), 0.14)
        self.assertEqual(run_js("D.stepHeight(0.31,1,0.04,5,0.1,0.32)"), 0.32)
        self.assertEqual(run_js("D.stepHeight(0.11,-1,0.04,5,0.1,0.32)"), 0.1)
        self.assertEqual(run_js("D.stepHeight(0.2,0,0.04,1,0.1,0.32)"), 0.2)

    def test_scale_by_preset_limits(self):
        self.assertEqual(run_js("D.scale([1,-1,0.5],{vx:0.25,vy:0.12,yaw_rate:0.8})"), [0.25, -0.12, 0.4])

    def test_detect_preset(self):
        presets = {"trot": {"height": 0.16, "period": 1.7}, "crawl": {"height": 0.2125, "period": 2.0}}
        self.assertEqual(run_js("D.detectPreset({height:0.2125,period:2.0},%s)" % json.dumps(presets)), "crawl")
        self.assertIsNone(run_js("D.detectPreset({height:0.3,period:1.0},%s)" % json.dumps(presets)))



FAKE_PAD = """
const D=require('../../site/assets/drive.js');
let tick=null; global.setInterval=f=>{tick=f;return 1;}; global.clearInterval=()=>{};
global.addEventListener=()=>{}; global.removeEventListener=()=>{};
const store={}; global.localStorage={getItem:k=>store[k]??null,setItem:(k,v)=>{store[k]=v;}};
let axes=%(rest)s, mapping=%(mapping)s;
Object.defineProperty(global,'navigator',{configurable:true,value:{getGamepads:()=>[{connected:true,id:'Android Pad',mapping,axes,buttons:Array.from({length:16},()=>({pressed:false}))}]}});
const sent=[]; const drive=D.create({meta:{drive:{trot:{vx:0.25,vy:0.12,yaw_rate:0.8}},pose:{body_roll:0.15,body_pitch:0.15,body_yaw:0.2},presets:{}},
  send:(p,b)=>sent.push(b),canvasActive:()=>true,reset(){},setPreset(){},toggleChase(){}});
const set=a=>{axes=a;tick();};
"""


@unittest.skipIf(NODE is None, "node is not installed")
class TestGamepadMapping(unittest.TestCase):
    def run_pad(self, rest, mapping, body):
        script = FAKE_PAD % {"rest": json.dumps(rest), "mapping": json.dumps(mapping)} + body
        return json.loads(subprocess.run([NODE, "-e", script], cwd=HERE, capture_output=True, text=True, check=True).stdout)

    def test_stick_axes_standard_and_non_standard(self):
        self.assertEqual(run_js("D.stickAxes([1,2,3,4],'standard',[0,0,0,0],null)"), [1, 2, 3, 4])
        # a trigger rests at -1 on axis 2: the right stick is the next axes that rested near zero
        self.assertEqual(run_js("D.stickAxes([0.1,0.2,-1,0.3,0.4,-1],'',[0,0,-1,0,0,-1],null)"), [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(run_js("D.stickAxes([0.5,0.6,0.7,0],'',[0,0,0,0],{lx:{i:3,sign:-1},ly:{i:2,sign:1},rx:{i:0,sign:1}})"), [-0, 0.7, 0.5, 0])

    def test_non_standard_pad_sends_commands(self):
        # forward on the left stick, and a right stick on axes 3 and 4 because axis 2 is a trigger resting at -1
        out = self.run_pad([0, 0, -1, 0, 0, -1], "''", """
            set([0,0,-1,0,0,-1]); set([0,-1,-1,0,0,-1]); set([0,-1,-1,0.9,0,-1]);
            console.log(JSON.stringify(sent.slice(-2).map(c=>[c.vx,c.yaw_rate,c.walk])));""")
        self.assertAlmostEqual(out[0][0], 0.25)
        self.assertEqual(out[0][1], 0)
        self.assertLess(out[1][1], 0)   # right stick right turns right (negative yaw rate)

    def test_rest_trigger_axis_is_not_a_turn_command(self):
        out = self.run_pad([0, 0, -1, 0, 0, -1], "''", "set([0,0,-1,0,0,-1]); console.log(JSON.stringify(sent.map(c=>c.yaw_rate)));")
        self.assertTrue(all(v == 0 for v in out))

    def test_calibration_learns_axes_and_signs(self):
        # this pad: left stick up = axis 5 going +1, left stick right = axis 4 going -1, right stick right = axis 1 going +1, right stick up = axis 2 going +1
        out = self.run_pad([0, 0, 0, 0, 0, 0], "''", """
            drive.calibrate(); set([0,0,0,0,0,0]);
            set([0,0,0,0,0,1]); set([0,0,0,0,0,0]);        // up on axis 5, release
            set([0,0,0,0,-1,0]); set([0,0,0,0,0,0]);       // right on axis 4 (negative), release
            set([0,1,0,0,0,0]); set([0,0,0,0,0,0]);        // right stick right on axis 1, release
            set([0,0,1,0,0,0]); set([0,0,0,0,0,0]);        // right stick up on axis 2, release
            const calibrated = drive.calib === null;
            set([0,0,0,0,0,1]); set([0,0,0,0,-1,1]); set([0,1,0,0,0,0]);
            console.log(JSON.stringify({calibrated, maps: drive.maps['Android Pad'], last: sent.slice(-3).map(c=>[c.vx,c.vy,c.yaw_rate])}));""")
        self.assertTrue(out["calibrated"])
        self.assertEqual(out["maps"]["ly"], {"i": 5, "sign": -1})
        self.assertEqual(out["maps"]["lx"], {"i": 4, "sign": -1})
        self.assertEqual(out["maps"]["rx"], {"i": 1, "sign": 1})
        self.assertEqual(out["maps"]["ry"], {"i": 2, "sign": -1})
        self.assertAlmostEqual(out["last"][0][0], 0.25)                       # forward
        self.assertAlmostEqual(out["last"][1][1], -0.12 / 2 ** 0.5, places=6)  # forward and right together: right is negative left, on the unit circle
        self.assertLess(out["last"][2][2], 0)

    def test_selected_pad_wins_over_the_one_that_moves(self):
        out = self.run_pad([0, 0, 0, 0], "'standard'", """
            const mk=(id,ax)=>({connected:true,id,index:id==='dead'?0:1,mapping:'standard',axes:ax,buttons:Array.from({length:16},()=>({pressed:false}))});
            let live=[0,0,0,0];
            Object.defineProperty(global,'navigator',{configurable:true,value:{getGamepads:()=>[mk('dead',[0,0,0,0]),mk('live',live)]}});
            tick(); live=[0,-1,0,0]; tick();
            const auto=drive.pad; drive.selectPad('dead'); tick();
            console.log(JSON.stringify({auto, chosen:drive.pad, list:drive.list.map(g=>[g.id,g.live]), saved:store['mojito-gamepad-choice'], last:sent[sent.length-1].vx}));""")
        self.assertEqual(out["auto"], "live")
        self.assertEqual(out["chosen"], "dead")
        self.assertEqual(out["list"], [["dead", False], ["live", True]])
        self.assertEqual(out["saved"], "dead")
        self.assertEqual(out["last"], 0)

    def test_nothing_is_sent_while_mapping(self):
        out = self.run_pad([0, 0, 0, 0], "'standard'", "drive.calibrate(); set([0,-1,0,0]); set([0,0,0,0]); console.log(JSON.stringify(sent));")
        self.assertEqual(out, [])



def run_touchpad_js(expression):
    script = "const T=require('../../site/assets/touchpad.js');console.log(JSON.stringify(%s));" % expression
    return json.loads(subprocess.run([NODE, "-e", script], cwd=HERE, capture_output=True, text=True, check=True).stdout)


@unittest.skipIf(NODE is None, "node is not installed")
class TestTouchpad(unittest.TestCase):
    def test_stick_vector_limits_to_unit_circle(self):
        radius = run_touchpad_js("T.RADIUS")
        self.assertEqual(run_touchpad_js("T.stickVector(0,0)"), [0, 0])
        self.assertEqual(run_touchpad_js("T.stickVector(%f,0)" % (radius / 2)), [0.5, 0])
        x, y = run_touchpad_js("T.stickVector(%f,%f)" % (3 * radius, -3 * radius))   # far past the edge, up and right
        self.assertAlmostEqual(x ** 2 + y ** 2, 1.0)
        self.assertAlmostEqual(x, -y)
        self.assertGreater(x, 0)
        self.assertLess(y, 0)   # up on the screen is negative, like a gamepad stick

    def test_touch_sticks_replace_the_gamepad_while_a_finger_is_down(self):
        script = FAKE_PAD % {"rest": "[0,0,0,0]", "mapping": "'standard'"}
        script = script.replace("send:(p,b)=>sent.push(b),", "touch:{active:()=>touching,axes:()=>[0,-1,0,0]},send:(p,b)=>sent.push(b),")
        script = "let touching=false;" + script + "set([0,0,0,0]); touching=true; set([0,0,0,0]); touching=false; set([0,0,0,0]);" \
            "console.log(JSON.stringify(sent.map(c=>c.vx)));"
        out = json.loads(subprocess.run([NODE, "-e", script], cwd=HERE, capture_output=True, text=True, check=True).stdout)
        self.assertEqual(out, [0, 0.25, 0])   # idle, finger up on the left stick, released



@unittest.skipIf(NODE is None, "node is not installed")
class TestDriveHeight(unittest.TestCase):
    """A held raise or lower input moves the stance height at drive's rate from the height the page reports, and only then sends one."""

    def run_height(self, body):
        script = FAKE_PAD % {"rest": "[0,0,0,0]", "mapping": "'standard'"}
        script = script.replace("yaw_rate:0.8}},", "yaw_rate:0.8,height_rate:0.04}},limits:{height:[0.1,0.32]},").replace("body_yaw:0.2}", "body_yaw:0.2,body_height:0.04}")
        script = "let clock=0;Object.defineProperty(performance,'now',{configurable:true,value:()=>clock});" + script + \
            "const tickFor=(n,axes)=>{for(let i=0;i<n;i++){clock+=33;set(axes||[0,0,0,0]);}};drive.sync({height:0.16});" + body
        return json.loads(subprocess.run([NODE, "-e", script], cwd=HERE, capture_output=True, text=True, check=True).stdout)

    def test_a_held_key_raises_the_height_at_the_rate(self):
        out = self.run_height("""tickFor(2); drive.keys.add('up'); tickFor(30); const raised=sent[sent.length-1].height;
            drive.keys.delete('up'); tickFor(2); const idle=sent[sent.length-1].height;
            console.log(JSON.stringify({raised, idle: idle===undefined}));""")
        self.assertAlmostEqual(out["raised"], 0.16 + 0.04 * 0.033 * 30, delta=0.002)   # 30 ticks of 33 ms at 0.04 m/s
        self.assertTrue(out["idle"])   # released: no height is sent, the slider is left alone

    def test_the_height_continues_from_where_it_was_left_and_clamps(self):
        out = self.run_height("""drive.keys.add('up'); tickFor(15); const first=sent[sent.length-1].height; drive.keys.delete('up'); tickFor(2);
            drive.sync({height:first}); drive.keys.add('up'); tickFor(15); const second=sent[sent.length-1].height;
            drive.keys.delete('up'); drive.keys.add('downh'); tickFor(2000); const low=sent[sent.length-1].height;
            console.log(JSON.stringify({first, second, low}));""")
        self.assertAlmostEqual(out["second"] - out["first"], 0.04 * 0.033 * 15, delta=0.002)
        self.assertEqual(out["low"], 0.1)   # the lower limit of the slider

    def test_right_stick_up_raises_and_pose_mode_leaves_the_height_alone(self):
        out = self.run_height("""tickFor(20,[0,0,0,-1]); const up=sent[sent.length-1].height; tickFor(2);
            drive.setMode('pose'); drive.keys.add('up'); tickFor(10); const pose=sent[sent.length-1];
            console.log(JSON.stringify({up, poseHasHeight: 'height' in pose, bodyHeight: pose.body_height}));""")
        self.assertGreater(out["up"], 0.16)
        self.assertFalse(out["poseHasHeight"])   # pose mode raises the body over the feet (body_height), not the stance
        self.assertGreater(out["bodyHeight"], 0)


if __name__ == "__main__":
    unittest.main()
