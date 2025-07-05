# wrench/core/registry.py

import inspect
from typing import Dict, Type, Union, Callable, Any, Literal, TypeVar
from functools import wraps
from .base import LabelModel, EndModel, JointModel

T = TypeVar("T")


class ComponentRegistry:
    """Registry for managing weak supervision components"""
    
    def __init__(self):
        self._registries = {
            'label': {},
            'end': {},
            'joint': {}
        }
        self._base_classes = {
            'label': LabelModel,
            'end': EndModel,
            'joint': JointModel
        }
        self._aliases: Dict[str, tuple] = {}  # alias -> (model_type, name)
    
    def _register_model(self, model_type: str, name: str, aliases: list = None):
        """Generic model registration decorator"""
        def decorator(cls: Type[T]) -> Type[T]:
            self._validate_model(cls, model_type, name)
            self._registries[model_type][name] = cls
            self._register_aliases(aliases, model_type, name)
            return cls
        return decorator
    
    def _validate_model(self, cls: Type, model_type: str, name: str):
        """Validate model class"""
        if not inspect.isclass(cls):
            raise TypeError(f"{model_type.title()} model '{name}' must be a class")
        
        base_class = self._base_classes[model_type]
        if not issubclass(cls, base_class):
            raise TypeError(f"{model_type.title()} model '{name}' must inherit from {base_class.__name__}")
        
        if name in self._registries[model_type]:
            raise ValueError(f"{model_type.title()} model '{name}' already registered")
        
        # Check required methods
        required_methods = ['fit', 'predict'] + (['transform'] if model_type == 'label' else [])
        missing = [m for m in required_methods if not hasattr(cls, m)]
        if missing:
            raise TypeError(f"{model_type.title()} model '{name}' missing methods: {missing}")
    
    def _register_aliases(self, aliases: list, model_type: str, name: str):
        """Register aliases for a model"""
        if not aliases:
            return
        
        for alias in aliases:
            if alias in self._aliases:
                raise ValueError(f"Alias '{alias}' already registered")
            self._aliases[alias] = (model_type, name)
    
    def _get_model(self, model_type: str, name: str, **kwargs):
        """Generic model getter"""
        # Resolve alias
        if name in self._aliases:
            resolved_type, resolved_name = self._aliases[name]
            if resolved_type != model_type:
                raise ValueError(f"'{name}' is an alias for {resolved_type} model, not {model_type}")
            name = resolved_name
        
        registry = self._registries[model_type]
        if name not in registry:
            available = list(registry.keys()) + self._get_aliases_for_type(model_type)
            raise ValueError(f"{model_type.title()} model '{name}' not found. Available: {available}")
        
        model_cls = registry[name]
        self._validate_kwargs(model_cls, kwargs, f"{model_type} model '{name}'")
        return model_cls(**kwargs)
    
    def _validate_kwargs(self, cls: Type, kwargs: dict, context: str):
        """Validate kwargs against class constructor"""
        try:
            sig = inspect.signature(cls.__init__)
            valid_params = {k: v for k, v in sig.parameters.items() if k != 'self'}
            
            # Check if constructor accepts **kwargs
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD 
                                 for p in sig.parameters.values())
            
            # Only validate parameter names if constructor doesn't accept **kwargs
            if not has_var_keyword:
                invalid = set(kwargs.keys()) - set(valid_params.keys())
                if invalid:
                    raise ValueError(f"Invalid parameters for {context}: {invalid}. "
                                   f"Valid: {list(valid_params.keys())}")
            
            # Check for missing required parameters (excluding **kwargs)
            required = [name for name, param in valid_params.items() 
                       if (param.default is inspect.Parameter.empty and 
                           param.kind != inspect.Parameter.VAR_KEYWORD)]
            missing = set(required) - set(kwargs.keys())
            if missing:
                raise ValueError(f"Missing required parameters for {context}: {missing}")
                
        except Exception as e:
            import warnings
            warnings.warn(f"Could not validate parameters for {context}: {e}")
    
    def _get_aliases_for_type(self, model_type: str) -> list:
        """Get aliases for a model type"""
        return [alias for alias, (mtype, _) in self._aliases.items() if mtype == model_type]
    
    # Public interface methods
    def register_label_model(self, name: str, aliases: list = None):
        return self._register_model('label', name, aliases)
    
    def register_end_model(self, name: str, aliases: list = None):
        return self._register_model('end', name, aliases)
    
    def register_joint_model(self, name: str, aliases: list = None):
        return self._register_model('joint', name, aliases)
    
    def get_label_model(self, name: str, **kwargs) -> LabelModel:
        return self._get_model('label', name, **kwargs)
    
    def get_end_model(self, name: str, **kwargs) -> EndModel:
        return self._get_model('end', name, **kwargs)
    
    def get_joint_model(self, name: str, **kwargs) -> JointModel:
        return self._get_model('joint', name, **kwargs)
    
    def list_models(self, model_type: str = None) -> dict:
        """List available models"""
        if model_type:
            return self._registries[model_type].copy()
        return {mtype: registry.copy() for mtype, registry in self._registries.items()}
    
    def list_aliases(self) -> Dict[str, tuple]:
        return self._aliases.copy()


# Global registry instance
_registry = ComponentRegistry()

# Decorator functions
def register_label_model(name: str, aliases: list = None):
    return _registry.register_label_model(name, aliases)

def register_end_model(name: str, aliases: list = None):
    return _registry.register_end_model(name, aliases)

def register_joint_model(name: str, aliases: list = None):
    return _registry.register_joint_model(name, aliases)

# Getter functions
def get_label_model(name: str, **kwargs) -> LabelModel:
    return _registry.get_label_model(name, **kwargs)

def get_end_model(name: str, **kwargs) -> EndModel:
    return _registry.get_end_model(name, **kwargs)

def get_joint_model(name: str, **kwargs) -> JointModel:
    return _registry.get_joint_model(name, **kwargs)

MODEL_TYPE = Literal['label', 'end', 'joint']

def create_model(model_type: MODEL_TYPE, name: str, **kwargs):
    """Generic model factory"""
    getters = {
        'label': get_label_model,
        'end': get_end_model,
        'joint': get_joint_model
    }
    if model_type not in getters:
        raise ValueError(f"Invalid model type: {model_type}. Must be one of {list(getters.keys())}")
    return getters[model_type](name, **kwargs)

def list_available_models() -> dict:
    """List all available models"""
    models = _registry.list_models()
    return {
        'label_models': list(models['label'].keys()),
        'end_models': list(models['end'].keys()),
        'joint_models': list(models['joint'].keys()),
        'aliases': _registry.list_aliases()
    }
    
    

def _ensure_models_loaded():
    """Ensure all model modules are imported to trigger registration"""
    if not hasattr(_ensure_models_loaded, '_loaded'):
        try:
            # Import all model modules to trigger registration decorators
            import skweak.labelmodel  # noqa: F401
            import skweak.endmodel #noqa: F401
        except ImportError as e:
            # If imports fail, try a more explicit approach
            import warnings
            warnings.warn(f"Could not auto-import models: {e}. "
                         f"Make sure to import model modules before using registry.")
        
        _ensure_models_loaded._loaded = True
        

