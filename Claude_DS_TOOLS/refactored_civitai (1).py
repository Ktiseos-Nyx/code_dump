# dataset_tools/vendored_sdpr/format/comfyui.py

__author__ = "receyuki"
__filename__ = "comfyui.py"
# MODIFIED by Ktiseos Nyx for Dataset-Tools
__copyright__ = "Copyright 2023, Receyuki; Modified 2025, Ktiseos Nyx"
__email__ = "receyuki@gmail.com; your_email@example.com"

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from .base_format import BaseFormat
from .utility import merge_dict


@dataclass
class ComfyUIConfig:
    """Configuration class for ComfyUI parsing - makes customization easy"""
    
    # Node type classifications for systematic traversal
    KSAMPLER_TYPES: Set[str] = field(default_factory=lambda: {
        "KSampler", "KSamplerAdvanced", "KSampler (Efficient)", "Efficient KSampler",
        "KSamplerSelect", "KSamplerCustom"  # Added common variants
    })
    
    VAE_ENCODE_TYPES: Set[str] = field(default_factory=lambda: {
        "VAEEncode", "VAEEncodeForInpaint", "VAEEncodeTiled"
    })
    
    CHECKPOINT_LOADER_TYPES: Set[str] = field(default_factory=lambda: {
        "CheckpointLoader", "CheckpointLoaderSimple", "unCLIPCheckpointLoader",
        "Checkpoint Loader (Simple)", "UltimateSDUpscaleLoader", "CheckpointLoaderNF4"
    })
    
    CLIP_TEXT_ENCODE_TYPES: Set[str] = field(default_factory=lambda: {
        "CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeSDXLRefiner",
        "smZ CLIPTextEncode", "BNK_CLIPTextEncoder", "CLIPTextEncodeFlux"
    })
    
    SAVE_IMAGE_TYPES: Set[str] = field(default_factory=lambda: {
        "SaveImage", "Image Save", "SDPromptSaver", "SaveImageWebsocket",
        "JWImageSave", "ImageSaveWebP"
    })
    
    PRIMITIVE_NODE_TYPES: Set[str] = field(default_factory=lambda: {
        "PrimitiveNode", "Primitive", "CR Seed", "Seed (Inspire)", "Seed (integer)",
        "BNK_INT", "IntegerPrimitive", "FloatPrimitive", "StringPrimitive"
    })

    # Parameter mapping - systematic approach to standardization
    PARAMETER_MAPPINGS: Dict[str, List[str]] = field(default_factory=lambda: {
        "model": ["ckpt_name", "checkpoint_name", "model_name"],
        "sampler_name": ["sampler_name", "sampler", "sampling_method"],
        "seed": ["seed", "noise_seed", "random_seed"],
        "cfg_scale": ["cfg", "cfg_scale", "guidance_scale"],
        "steps": ["steps", "sampling_steps", "num_steps"],
        "scheduler": ["scheduler", "schedule", "noise_schedule"],
        "denoise": ["denoise", "denoising_strength", "noise_strength"],
        "width": ["width", "image_width", "w"],
        "height": ["height", "image_height", "h"]
    })
    
    # ComfyUI-specific settings to capture
    COMFY_SPECIFIC_KEYS: Set[str] = field(default_factory=lambda: {
        "add_noise", "start_at_step", "end_at_step", "return_with_left_over_noise",
        "upscale_method", "upscaler", "lora_name", "lora_strength", "control_net",
        "controlnet_conditioning_scale", "vae_name", "clip_skip"
    })
    
    # Input traversal priority for unknown nodes
    TRAVERSAL_PRIORITY: List[str] = field(default_factory=lambda: [
        "model", "clip", "conditioning", "positive", "negative", "latent_image",
        "samples", "image", "pixels", "vae", "control_net", "source_image"
    ])


class NodeTraversalState:
    """Encapsulates state during node traversal to prevent pollution of main class"""
    
    def __init__(self):
        self.positive_prompt: str = ""
        self.negative_prompt: str = ""
        self.positive_sdxl: Dict[str, str] = {}
        self.negative_sdxl: Dict[str, str] = {}
        self.is_sdxl: bool = False
        self.visited_nodes: Set[str] = set()
        self.path_history: List[str] = []
        
    def reset_for_new_traversal(self):
        """Reset state for a new traversal path"""
        self.positive_prompt = ""
        self.negative_prompt = ""
        self.positive_sdxl = {}
        self.negative_sdxl = {}
        self.is_sdxl = False
        self.visited_nodes = set()
        self.path_history = []


@dataclass
class FlowExtractionResult:
    """Structured result from workflow traversal"""
    positive_prompt: str = ""
    negative_prompt: str = ""
    positive_sdxl: Dict[str, str] = field(default_factory=dict)
    negative_sdxl: Dict[str, str] = field(default_factory=dict)
    is_sdxl: bool = False
    parameters: Dict[str, str] = field(default_factory=dict)
    custom_settings: Dict[str, str] = field(default_factory=dict)
    width: str = "0"
    height: str = "0"
    
    def get_meaningful_param_count(self) -> int:
        """Calculate how many meaningful parameters this result contains"""
        count = 0
        
        # Count prompts
        if self.positive_prompt or self.positive_sdxl:
            count += 1
        if self.negative_prompt or self.negative_sdxl:
            count += 1
            
        # Count standard parameters (excluding placeholders)
        for value in self.parameters.values():
            if value and value != BaseFormat.DEFAULT_PARAMETER_PLACEHOLDER:
                count += 1
                
        # Count dimensions
        if self.width and self.width != "0":
            count += 1
        if self.height and self.height != "0":
            count += 1
            
        return count


class ComfyUINodeProcessor:
    """Handles processing of individual node types - separation of concerns"""
    
    def __init__(self, config: ComfyUIConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def process_ksampler_node(self, node_data: Dict[str, Any], state: NodeTraversalState) -> Dict[str, Any]:
        """Extract parameters from KSampler-type nodes"""
        inputs = node_data.get("inputs", {})
        flow_data = {}
        
        # Direct parameter extraction
        direct_params = [
            "seed", "noise_seed", "steps", "cfg", "sampler_name", "scheduler",
            "denoise", "add_noise", "start_at_step", "end_at_step", "return_with_left_over_noise"
        ]
        
        for param in direct_params:
            if param in inputs and inputs[param] is not None:
                flow_data[param] = inputs[param]
                
        return flow_data
        
    def process_clip_encode_node(self, node_data: Dict[str, Any], node_id: str, 
                               workflow_data: Dict[str, Any], state: NodeTraversalState) -> Union[str, Dict[str, str]]:
        """Process CLIP text encoding nodes"""
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        
        # Handle different CLIP encoder types
        if class_type in ["CLIPTextEncode", "smZ CLIPTextEncode"]:
            return self._extract_text_from_input(inputs.get("text"), workflow_data, state)
            
        elif class_type == "CLIPTextEncodeSDXL":
            state.is_sdxl = True
            sdxl_prompts = {}
            
            for clip_type, input_key in [("G", "text_g"), ("L", "text_l")]:
                text = self._extract_text_from_input(inputs.get(input_key), workflow_data, state)
                sdxl_prompts[f"Clip {clip_type}"] = text.strip() if isinstance(text, str) else ""
                
            return sdxl_prompts
            
        elif class_type == "CLIPTextEncodeSDXLRefiner":
            state.is_sdxl = True
            text = self._extract_text_from_input(inputs.get("text"), workflow_data, state)
            return {"Refiner": text.strip() if isinstance(text, str) else ""}
            
        return ""
        
    def _extract_text_from_input(self, text_input: Any, workflow_data: Dict[str, Any], 
                                state: NodeTraversalState) -> str:
        """Extract text from either direct input or linked node"""
        if isinstance(text_input, list) and len(text_input) >= 1 and text_input[0] is not None:
            # Text is linked from another node
            linked_node_id = str(text_input[0])
            if linked_node_id in workflow_data and linked_node_id not in state.visited_nodes:
                linked_node = workflow_data[linked_node_id]
                
                # Handle primitive nodes
                if linked_node.get("class_type") in self.config.PRIMITIVE_NODE_TYPES:
                    widgets = linked_node.get("widgets_values", [])
                    if widgets:
                        return str(widgets[0])
                        
        elif isinstance(text_input, str):
            return text_input
            
        return ""


class ComfyUI(BaseFormat):
    """
    Enhanced ComfyUI format parser with systematic approach to workflow traversal.
    Engineered for maintainability, extensibility, and robustness.
    """
    
    tool = "ComfyUI"
    
    def __init__(
        self,
        info: Optional[Dict[str, Any]] = None,
        raw: str = "",
        width: Any = 0,
        height: Any = 0,
        logger_obj: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        super().__init__(
            info=info,
            raw=raw,
            width=width,
            height=height,
            logger_obj=logger_obj,
            **kwargs,
        )
        
        # Configuration and processors
        self.config = ComfyUIConfig()
        self.node_processor = ComfyUINodeProcessor(self.config, self._logger)
        
        # Workflow data containers
        self._prompt_json: Dict[str, Any] = {}
        self._workflow_json: Optional[Dict[str, Any]] = None
        
        # Traversal state
        self._traversal_state = NodeTraversalState()

    def _process(self) -> None:
        """Main processing pipeline for ComfyUI workflows"""
        self._logger.debug(f"{self.tool}: Starting ComfyUI workflow processing")
        
        # Extract and validate workflow data
        if not self._extract_workflow_data():
            return
            
        # Find and evaluate possible end nodes
        end_node_candidates = self._find_end_node_candidates()
        if not end_node_candidates:
            self._logger.warning(f"{self.tool}: No suitable end nodes found for traversal")
            self.status = self.Status.COMFYUI_ERROR
            self._error = "No SaveImage or KSampler nodes found in workflow"
            return
            
        # Find the best traversal path
        best_result = self._evaluate_traversal_paths(end_node_candidates)
        if not best_result or best_result.get_meaningful_param_count() == 0:
            self._logger.warning(f"{self.tool}: No meaningful data extracted from workflow")
            self.status = self.Status.COMFYUI_ERROR
            self._error = "Workflow traversal yielded no meaningful data"
            return
            
        # Apply the best result to self
        self._apply_extraction_result(best_result)
        
        self._logger.info(f"{self.tool}: Successfully extracted workflow data")

    def _extract_workflow_data(self) -> bool:
        """Extract and validate workflow JSON from PNG chunks"""
        prompt_str = str(self._info.get("prompt", ""))
        workflow_str = str(self._info.get("workflow", ""))
        
        # Determine source and parse
        source_data, source_desc = self._select_workflow_source(prompt_str, workflow_str)
        if not source_data:
            self._logger.warning(f"{self.tool}: No workflow data found in PNG chunks")
            self.status = self.Status.MISSING_INFO
            self._error = "No ComfyUI workflow data found in PNG metadata"
            return False
            
        try:
            parsed_workflow = json.loads(source_data)
            if not isinstance(parsed_workflow, dict):
                raise ValueError("Workflow data is not a dictionary")
                
            # Store based on structure
            if self._is_api_format(parsed_workflow):
                self._workflow_json = parsed_workflow
                self._logger.debug(f"{self.tool}: Loaded API format workflow")
            else:
                self._prompt_json = parsed_workflow
                self._logger.debug(f"{self.tool}: Loaded standard workflow")
                
            # Set raw data if not already set
            if not self._raw:
                self._raw = source_data
                
            return True
            
        except (json.JSONDecodeError, ValueError) as e:
            self._logger.error(f"{self.tool}: Failed to parse workflow JSON: {e}")
            self.status = self.Status.FORMAT_ERROR
            self._error = f"Invalid JSON in ComfyUI workflow: {e}"
            return False

    def _select_workflow_source(self, prompt_str: str, workflow_str: str) -> Tuple[str, str]:
        """Select the best workflow source from available data"""
        if prompt_str.strip():
            return prompt_str, "PNG 'prompt' chunk"
        elif workflow_str.strip():
            return workflow_str, "PNG 'workflow' chunk"
        else:
            return "", ""

    def _is_api_format(self, workflow_data: Dict[str, Any]) -> bool:
        """Determine if workflow is in API format (has nodes/links) vs standard format"""
        return "nodes" in workflow_data and "links" in workflow_data

    def _find_end_node_candidates(self) -> Dict[str, str]:
        """Find suitable end nodes for workflow traversal"""
        workflow_data = self._prompt_json or self._workflow_json
        if not workflow_data:
            return {}
            
        candidates = {}
        
        # Check for SaveImage nodes first (preferred end points)
        save_nodes = self._find_nodes_by_type(workflow_data, self.config.SAVE_IMAGE_TYPES)
        if save_nodes:
            candidates.update(save_nodes)
            self._logger.debug(f"{self.tool}: Found {len(save_nodes)} SaveImage candidates")
        else:
            # Fallback to KSampler nodes
            ksampler_nodes = self._find_nodes_by_type(workflow_data, self.config.KSAMPLER_TYPES)
            candidates.update(ksampler_nodes)
            self._logger.debug(f"{self.tool}: Found {len(ksampler_nodes)} KSampler candidates")
            
        return candidates

    def _find_nodes_by_type(self, workflow_data: Dict[str, Any], node_types: Set[str]) -> Dict[str, str]:
        """Find all nodes matching the specified types"""
        found_nodes = {}
        
        for node_id, node_data in workflow_data.items():
            if not isinstance(node_data, dict):
                continue
                
            class_type = node_data.get("class_type", "")
            if class_type in node_types:
                found_nodes[node_id] = class_type
                
        return found_nodes

    def _evaluate_traversal_paths(self, end_candidates: Dict[str, str]) -> Optional[FlowExtractionResult]:
        """Evaluate all possible traversal paths and return the best one"""
        best_result = None
        max_param_count = -1
        
        self._logger.debug(f"{self.tool}: Evaluating {len(end_candidates)} traversal paths")
        
        for node_id, class_type in end_candidates.items():
            self._logger.debug(f"{self.tool}: Traversing from {node_id} ({class_type})")
            
            # Reset state for this traversal
            self._traversal_state.reset_for_new_traversal()
            
            # Run traversal
            result = self._traverse_from_node(node_id)
            param_count = result.get_meaningful_param_count()
            
            self._logger.debug(f"{self.tool}: Path from {node_id} yielded {param_count} parameters")
            
            if param_count > max_param_count:
                max_param_count = param_count
                best_result = result
                
        if best_result:
            self._logger.info(f"{self.tool}: Best path selected with {max_param_count} parameters")
            
        return best_result

    def _traverse_from_node(self, start_node_id: str) -> FlowExtractionResult:
        """Traverse workflow from a given starting node"""
        workflow_data = self._prompt_json or self._workflow_json
        if not workflow_data:
            return FlowExtractionResult()
            
        # Reset traversal state
        self._traversal_state.reset_for_new_traversal()
        
        # Run the traversal
        flow_data, _ = self._traverse_node_recursive(workflow_data, start_node_id)
        
        # Convert to structured result
        result = self._convert_flow_to_result(flow_data)
        
        # Apply prompts from traversal state
        result.positive_prompt = self._traversal_state.positive_prompt
        result.negative_prompt = self._traversal_state.negative_prompt
        result.positive_sdxl = self._traversal_state.positive_sdxl.copy()
        result.negative_sdxl = self._traversal_state.negative_sdxl.copy()
        result.is_sdxl = self._traversal_state.is_sdxl
        
        return result

    def _traverse_node_recursive(self, workflow_data: Dict[str, Any], node_id: str) -> Tuple[Dict[str, Any], List[str]]:
        """Recursively traverse workflow nodes"""
        if node_id in self._traversal_state.visited_nodes:
            return {}, []  # Prevent infinite loops
            
        if node_id not in workflow_data:
            self._logger.warning(f"{self.tool}: Node {node_id} not found in workflow")
            return {}, []
            
        self._traversal_state.visited_nodes.add(node_id)
        self._traversal_state.path_history.append(node_id)
        
        node_data = workflow_data[node_id]
        if not isinstance(node_data, dict):
            return {}, []
            
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        
        flow_data = {}
        
        # Process based on node type
        flow_data.update(self._process_node_by_type(node_data, node_id, workflow_data, class_type))
        
        # Traverse connected nodes
        flow_data.update(self._traverse_connected_nodes(inputs, workflow_data, class_type))
        
        return flow_data, self._traversal_state.path_history.copy()

    def _process_node_by_type(self, node_data: Dict[str, Any], node_id: str, 
                             workflow_data: Dict[str, Any], class_type: str) -> Dict[str, Any]:
        """Process node based on its type"""
        flow_data = {}
        
        if class_type in self.config.KSAMPLER_TYPES:
            flow_data.update(self.node_processor.process_ksampler_node(node_data, self._traversal_state))
            flow_data.update(self._extract_latent_dimensions(node_data, workflow_data))
            
        elif class_type in self.config.CLIP_TEXT_ENCODE_TYPES:
            text_result = self.node_processor.process_clip_encode_node(
                node_data, node_id, workflow_data, self._traversal_state
            )
            self._apply_clip_text_result(text_result, class_type)
            
        elif class_type in self.config.CHECKPOINT_LOADER_TYPES:
            inputs = node_data.get("inputs", {})
            if "ckpt_name" in inputs:
                flow_data["ckpt_name"] = inputs["ckpt_name"]
                
        elif class_type == "LoraLoader":
            inputs = node_data.get("inputs", {})
            if "lora_name" in inputs:
                flow_data["lora_name"] = inputs["lora_name"]
                
        return flow_data

    def _extract_latent_dimensions(self, ksampler_node: Dict[str, Any], workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dimensions from connected EmptyLatentImage node"""
        inputs = ksampler_node.get("inputs", {})
        latent_link = inputs.get("latent_image")
        
        if not isinstance(latent_link, list) or len(latent_link) < 1:
            return {}
            
        latent_node_id = str(latent_link[0])
        if latent_node_id not in workflow_data:
            return {}
            
        latent_node = workflow_data[latent_node_id]
        if latent_node.get("class_type") != "EmptyLatentImage":
            return {}
            
        latent_inputs = latent_node.get("inputs", {})
        dims = {}
        
        if "width" in latent_inputs:
            dims["k_width"] = latent_inputs["width"]
        if "height" in latent_inputs:
            dims["k_height"] = latent_inputs["height"]
            
        return dims

    def _apply_clip_text_result(self, text_result: Union[str, Dict[str, str]], class_type: str):
        """Apply CLIP text encoding results to traversal state"""
        if isinstance(text_result, str):
            # Determine if this should be positive or negative based on context
            # This is a simplification - in practice you'd need more context
            if not self._traversal_state.positive_prompt:
                self._traversal_state.positive_prompt = text_result
            elif not self._traversal_state.negative_prompt:
                self._traversal_state.negative_prompt = text_result
                
        elif isinstance(text_result, dict):
            if "Clip G" in text_result or "Clip L" in text_result:
                self._traversal_state.positive_sdxl.update(text_result)
            elif "Refiner" in text_result:
                self._traversal_state.positive_sdxl.update(text_result)

    def _traverse_connected_nodes(self, inputs: Dict[str, Any], workflow_data: Dict[str, Any], 
                                 class_type: str) -> Dict[str, Any]:
        """Traverse nodes connected to current node's inputs"""
        flow_data = {}
        
        # Get traversal priority based on node type
        input_priority = self._get_input_priority(class_type)
        
        for input_name in input_priority:
            if input_name not in inputs:
                continue
                
            input_link = inputs[input_name]
            if not isinstance(input_link, list) or len(input_link) < 1:
                continue
                
            connected_node_id = str(input_link[0])
            if connected_node_id in self._traversal_state.visited_nodes:
                continue
                
            sub_flow, _ = self._traverse_node_recursive(workflow_data, connected_node_id)
            flow_data = merge_dict(flow_data, sub_flow)
            
        return flow_data

    def _get_input_priority(self, class_type: str) -> List[str]:
        """Get input traversal priority based on node type"""
        if class_type in self.config.KSAMPLER_TYPES:
            return ["model", "positive", "negative", "latent_image"]
        elif class_type in self.config.SAVE_IMAGE_TYPES:
            return ["images"]
        else:
            return self.config.TRAVERSAL_PRIORITY

    def _convert_flow_to_result(self, flow_data: Dict[str, Any]) -> FlowExtractionResult:
        """Convert raw flow data to structured result"""
        result = FlowExtractionResult()
        
        # Extract standardized parameters
        handled_keys = set()
        for standard_key, possible_keys in self.config.PARAMETER_MAPPINGS.items():
            for key in possible_keys:
                if key in flow_data and flow_data[key] is not None:
                    result.parameters[standard_key] = self._clean_parameter_value(flow_data[key])
                    handled_keys.add(key)
                    break
                    
        # Handle dimensions specially
        if "k_width" in flow_data:
            result.width = str(flow_data["k_width"])
            handled_keys.add("k_width")
        if "k_height" in flow_data:
            result.height = str(flow_data["k_height"])
            handled_keys.add("k_height")
            
        # Add ComfyUI-specific settings
        for key in self.config.COMFY_SPECIFIC_KEYS:
            if key in flow_data and key not in handled_keys:
                display_key = self._format_key_for_display(key)
                result.custom_settings[display_key] = self._clean_parameter_value(flow_data[key])
                handled_keys.add(key)
                
        # Add remaining unhandled parameters
        for key, value in flow_data.items():
            if key not in handled_keys and value is not None:
                display_key = self._format_key_for_display(key)
                result.custom_settings[display_key] = self._clean_parameter_value(value)
                
        return result

    def _apply_extraction_result(self, result: FlowExtractionResult):
        """Apply extraction result to self attributes"""
        self._positive = result.positive_prompt
        self._negative = result.negative_prompt
        self._positive_sdxl = result.positive_sdxl
        self._negative_sdxl = result.negative_sdxl
        self._is_sdxl = result.is_sdxl
        
        # Handle SDXL prompt merging if needed
        if self._is_sdxl and not self._positive and self._positive_sdxl:
            self._positive = self.merge_clip(self._positive_sdxl)
        if self._is_sdxl and not self._negative and self._negative_sdxl:
            self._negative = self.merge_clip(self._negative_sdxl)
            
        # Apply parameters
        self._parameter.update(result.parameters)
        
        # Apply dimensions
        if result.width != "0":
            self._width = result.width
            self._parameter["width"] = result.width
        if result.height != "0":
            self._height = result.height
            self._parameter["height"] = result.height
        if result.width != "0" and result.height != "0":
            self._parameter["size"] = f"{result.width}x{result.height}"
            
        # Build settings string
        self._setting = self._build_settings_string(
            custom_settings_dict=result.custom_settings,
            include_standard_params=True,
            sort_parts=True,
        )

    def _clean_parameter_value(self, value: Any) -> str:
        """Clean and standardize parameter values"""
        value_str = str(value).strip()
        
        # Remove quotes if they wrap the entire string
        if len(value_str) >= 2:
            if (value_str.startswith('"') and value_str.endswith('"')) or \
               (value_str.startswith("'") and value_str.endswith("'")):
                return value_str[1:-1]
                
        return value_str

    @staticmethod
    def merge_clip(data: Dict[str, str]) -> str:
        """Merge SDXL CLIP G and L prompts intelligently"""
        clip_g = str(data.get("Clip G", "")).strip(" ,")
        clip_l = str(data.get("Clip L", "")).strip(" ,")
        
        if not clip_g and not clip_l:
            return ""
        if clip_g == clip_l:
            return clip_g
        if not clip_g:
            return clip_l
        if not clip_l:
            return clip_g
            
        return f"Clip G: {clip_g}, Clip L: {clip_l}"

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get detailed information about the parsed workflow"""
        workflow_data = self._prompt_json or self._workflow_json
        
        info = {
            "workflow_format": "API" if self._workflow_json else "Standard",
            "total_nodes": len(workflow_data) if workflow_data else 0,
            "has_save_nodes": bool(self._find_nodes_by_type(workflow_data or {}, self.config.SAVE_IMAGE_TYPES)),
            "has_ksampler_nodes": bool(self._find_nodes_by_type(workflow_data or {}, self.config.KSAMPLER_TYPES)),
            "parameter_count": len(self._parameter),
            "is_sdxl": self._is_sdxl,
        }
        
        if workflow_data:
            # Count node types
            node_types = {}
            for node_data in workflow_data.values():
                if isinstance(node_data, dict):
                    class_type = node_data.get("class_type", "Unknown")
                    node_types[class_type] = node_types.get(class_type, 0) + 1
            info["node_type_counts"] = node_types
            
        return info