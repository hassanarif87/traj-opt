# Optimizing spacecraft trajectories 

Sandbox for prototyping trajectory optimization formulations 

# CLI Commands

Run the two-stage ascent optimization and write its trajectory CSV to `output/`:

```bash
uv run topt run --scenario scenarios/two_stage_ascent.yaml
```

Generate an HTML report from the resulting trajectory:

```bash
uv run topt report --scenario two_stage_ascent --output two_stage_ascent
```

View the [example two-stage ascent report](two_stage_ascent.html).

# Transcriptions Methods
## MultiShootingTranscription
A class to transcribe a multi-phase trajectory optimization problem into a Nonlinear Programming (NLP) problem using multiple shooting. Where each phase is a shooting segment

# Examples 
## Shooting method
 - Single stage to orbit lunar ascent using Linear Tangent Steering 
## Collocation method
 - Single stage to orbit lunar ascent
 - Cart pole 

## Multiple shooting 
 - Multiple shooting cannon 
 - Two degree of freedom Two stage ascent with ball-park aero, exponential atmosphere.
 - Second Stage ascent, exo-atmospheric
 
## References 
- Optimal Control with Aerospace Applications (Space Technology Library) by Longuski, James M., Guzmán, Jose J., Prussing, John E.(November 5, 2013)
- An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation- Kelly, Matthew 
- Missile Aerodynamics for Ascent and Re-entry - GL Watts · 2012 
- Robert A. Braeunig, 2020. "Ballpark" CD - <http://www.braeunig.us/space/aerodyn_wip.htm> 
- A Convex Approach to Rocket Ascent Trajectory Optimization - B Benedikter · 2019
- Vega Launchers’ Trajectory Optimization Using a Pseudospectral Transcription- GDCB de Volo · 2019


