
select 
p.datanascimento, lsri.* from vw_Cad_LancamentoServicoResultadoItem lsri
left join vw_cad_paciente p on p.matriculal = lsri.matricula and lsri.paciente = p.nome
where (0=0)
and (LEN(lsri.exameResultado) > 0)
and paciente = ? and p.DataNascimento = ?
and lsri.desativado = 0
order by lsri.grupo, lsri.servico, lsri.grupopagina, lsri.grupopaginaordem
