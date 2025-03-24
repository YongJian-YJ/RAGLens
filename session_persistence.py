import streamlit as st
import json
import os


class PersistentStorage:
    def __init__(self, filename="app_state.json"):
        self.filename = filename
        self.default_values = {
            "chunk_size": 2000,  # default value
            "chunk_overlap": 200,  # default value
            "active_directory": "default",  # default value
            "llm_type": "local",
        }

    def load_state(self):
        """Load state from JSON file or create with defaults if doesn't exist"""
        try:
            if os.path.exists(self.filename):
                with open(self.filename, "r") as f:
                    saved_state = json.load(f)
                    # Update session state with saved values
                    for key in self.default_values:
                        st.session_state[key] = saved_state.get(
                            key, self.default_values[key]
                        )
            else:
                # Initialize with default values if file doesn't exist
                for key, value in self.default_values.items():
                    st.session_state[key] = value
                self.save_state()
        except Exception as e:
            st.error(f"Error loading state: {str(e)}")

    def save_state(self):
        """Save current values of the three variables to JSON file"""
        try:
            state_to_save = {
                "chunk_size": st.session_state.chunk_size,
                "chunk_overlap": st.session_state.chunk_overlap,
                "active_directory": st.session_state.active_directory,
                "llm_type": st.session_state.llm_type,
            }
            with open(self.filename, "w") as f:
                json.dump(state_to_save, f, indent=4)
        except Exception as e:
            st.error(f"Error saving state: {str(e)}")


# Example usage in your Streamlit app
def main():
    # Initialize storage
    storage = PersistentStorage()

    # Load saved state at startup
    storage.load_state()

    # Create input widgets
    st.number_input(
        "Chunk Size",
        value=st.session_state.chunk_size,
        key="chunk_size",
        on_change=storage.save_state,
    )

    st.number_input(
        "Chunk Overlap",
        value=st.session_state.chunk_overlap,
        key="chunk_overlap",
        on_change=storage.save_state,
    )

    st.text_input(
        "Active Directory",
        value=st.session_state.active_directory,
        key="active_directory",
        on_change=storage.save_state,
    )


if __name__ == "__main__":
    main()
