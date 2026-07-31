use crate::burn_model::Model;
use burn::{module::Module, prelude::Device, store::ModuleRecord, tensor::Bytes};

static STATE_ENCODED: &[u8] = include_bytes!("../model.bpk");

/// 构建并加载训练好的模型参数
pub fn build_and_load_model() -> (Model, Device) {
    let device = Device::default();
    let model = Model::new(&device);

    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(STATE_ENCODED.to_vec()))
        .expect("Failed to decode burnpack model.bpk state");

    (model.load_record(record), device)
}
