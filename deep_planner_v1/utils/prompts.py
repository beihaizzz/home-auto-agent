scheme_plan_prompt = """You are an AI assistant tasked with generating a list of smart home device usage scenarios 
based on the current date and device configuration. Your goal is to create realistic and varied scenarios that 
showcase how a user might interact with their smart home devices throughout the day.

You will be provided with two inputs:

<date>
{date}
</date>


<device_configs>
{device_configs}
</device_configs>

Using this information, generate a list of smart home usage scenarios. Consider the following factors when creating 
these scenarios:

1. The current date and potential seasonal activities or weather conditions in <location>{location}<location>
2. The time of day for each scenario (morning, afternoon, evening, night)
3. Typical daily routines (e.g., waking up, going to work, returning home, preparing for bed)
4. The available devices and their functions as specified in the device configuration
5. Realistic use cases for each device
6. Potential combinations of device usage for more complex scenarios
7. Description must in chinese

Format your output using the following class structures:

```python
class Scene(BaseModel):
    name: str = Field(
        description="Name of the Scene",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this Scene.",
    )
    research: bool = Field(
        description="Whether to perform web research for this Scene of the report."
    )
    start_time: time = Field(
        description="When the scene might happened"
    )
    end_time: time = Field(
        description="When the scene might ended"
    )
    involved_devices: List[Device] = Field(
        description="Devices involved in this scene"
    )
    
class Device(BaseModel):
    id: str = Field(
        description="Id of the devices will be used in this scene"
    )
    type: str = Field(
        description="Type of the devices will be used in this scene,such as ac and led light"
    )
    config: Dict[str, Any] | None = Field(
        description="The corresponding config in device_configs for the corresponding device_id",
        default=None
    )

class Scenes(BaseModel):
    scenes: List[Scene] = Field(
        description="Scene of the scheme today",
    )
```

Examples of possible scenarios:
1. Morning routine: Waking up, turning on lights, adjusting thermostat, starting coffee maker
2. Leaving for work: Turning off lights, setting security system, adjusting thermostat for energy savings
3. Returning home: Disarming security system, turning on lights, adjusting thermostat, starting robot vacuum
4. Evening relaxation: Dimming lights, playing music, adjusting air purifier settings
5. Bedtime routine: Turning off lights, setting security system, adjusting thermostat for sleep

Generate 5-7 scenarios that cover different parts of the day and utilize various devices from the configuration. 
Ensure that the scenarios are realistic and take into account the current date.

Your final output should only include the Scenes object with its list of Scene objects, formatted as Python code. Do 
not include any additional explanations or comments in your response."""

query_gen_prompt = """You are an AI assistant tasked with generating web search queries based on smart home usage 
scenes. Your goal is to analyze the provided scene and create appropriate search queries to retrieve external 
information that enhances the smart home experience, such as weather data, device optimization tips, 
or context-specific details.

You will be given a smart home usage scene in the following format:

<scene>
{scene}
</scene>

<location>
{location}
<location>

<device_configs> {device_configs} </device_configs> Using this information, generate a series of search queries to 
gather relevant information for the scene. When creating these queries, consider the following factors:

1. **Time** - The scene's start and end times, and how they influence potential needs. For example, morning might 
require weather forecasts, while evening might need relaxation tips.

2. **Involved Devices** - The devices listed in the scene (refer to `involved_devices`) and any external information 
that could optimize their use. For example, temperature data for an air conditioner or lighting intensity for an LED 
lamp.

3. **Weather or Environment** - Conditions that might affect device usage or user comfort, such as temperature, 
humidity, or other environmental factors.

4. **Contextual Needs** - Additional information related to the scene’s activities. For example, traffic updates for 
a "leaving home" scene or sleep advice for a "bedtime" scene.

### Two Main Objectives for Query Generation - **Objective 1: Complete Device Parameter Information** Retrieve 
external information to provide necessary data for device control. For instance, an air conditioner might need 
queries like "today’s weather temperature and humidity," while an LED lamp might need "optimal lighting parameters 
for a specific time."

- **Objective 2: Enhance Scene Integration** Consider how to humanize device control to better align with user needs. 
For example, a "night light" scene might require queries like "most comfortable lighting intensity at night" or 
"impact of nighttime lighting on sleep."

### Examples
1. **Scene: Turning on a Night Light**
   - Query 1: `"most comfortable lighting intensity at night"`
   - Query 2: `"impact of nighttime lighting on sleep"`

2. **Scene: Humidifier in Humid Weather (e.g., Southern China’s ‘Hui Nan Tian’)**  
   - Query 1: `"typical humidity levels during Hui Nan Tian"`
   - Query 2: `"best humidifier settings in high humidity environments"`

### Output Format
Format your output using the following class structure:

```python
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="The query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(description="A list of search queries.")"""

scheme_gen_prompt = """You are an AI agent tasked with generating a detailed scheme for controlling a group of 
devices based on a given scene and relevant information. Your goal is to create a structured output that includes 
device calls with appropriate parameters and order.

Input variables:
<scene>
{scene}
</scene>

This variable contains information about the scene, including its name, description, start and end times, 
and involved devices.

<source_strs>
{source_strs}
</source_strs>

This variable defines the structure of the output you need to generate.

Instructions: 1. Carefully analyze the scene information, paying attention to the name, description, start and end 
times, and involved devices.

2. Review the source_strs to gather relevant information about optimal device settings, environmental factors, 
and best practices for the given scene.

3. For each device involved in the scene, create a DeviceCall object with the following information: a. device_name: 
The name or type of the device b. device_id: The ID of the device as provided in the scene information c. params: A 
dictionary of parameters for the device, based on the scene requirements and information from source_strs d. order: 
The order in which the device should be activated or adjusted in the scene (starting from 0)

4. Consider the logical sequence of device activations or adjustments when assigning the order value to each DeviceCall.

5. Ensure that the parameters for each device are appropriate for the scene and align with the information provided 
in source_strs. For example, if the scene involves sleep, use the information about optimal light colors and 
intensities for sleep when setting parameters for lighting devices.

6. If the scene involves temperature control, use the weather information provided in source_strs to inform your 
decisions about air conditioning or heating parameters.

7. Create a Scheme object that includes:
   a. device_calls: A list of all the DeviceCall objects you've created
   b. scene: The original scene object provided in the input

8. Structure your output as a JSON-like representation of the Scheme object, ensuring that all required fields are 
included and properly formatted.

Your final output should only include the structured Scheme object, without any additional explanation or commentary. 
Use the following format for your output:

<output>
```python
class DeviceCall(BaseModel, Generic[ConfigT]):
    device_name: str
    device_id: str
    config: ConfigT = Field(
        description="the params for device_call which comes from the device_configs",
    )
    order: int = Field(
        description="The order of the device call in the scene.",
    )


class Scheme(BaseModel,Generic[ConfigT]):
    device_calls: List[DeviceCall[ConfigT]]
    scene: Scene
```

</output>

Ensure that your output is a valid JSON structure and includes all necessary information from the input scene and 
your generated device calls."""
